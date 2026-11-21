#include <catch2/catch_all.hpp>
#include <algorithm>
#include <utility>
#include <vector>
#include "libslic3r/TriangleMesh.hpp"
#include "libslic3r/Polygon.hpp"
#include "libslic3r/BoundingBox.hpp"
#include "libslic3r/Point.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "libslic3r/MeshRepair.hpp"
#include "libslic3r/Model.hpp"
#include "libslic3r/ObjectID.hpp"
#include "libslic3r/Geometry.hpp"
#include "libslic3r/libslic3r.h"

using namespace Slic3r;

// convex_hull_2d does not clip geometry below the bed, so these cases avoid
// sinking transforms.
TEST_CASE("A part's 2D convex hull is its footprint projected onto the bed", "[Model]")
{
    Model model;
    ModelObject* object = model.add_object();
    // Keep the cube's raw coordinates ([0,20] on every axis): the default
    // add_volume re-centers the geometry, which would move the footprint.
    object->add_volume(make_cube(20, 20, 20), ModelVolumeType::MODEL_PART, false);

    SECTION("identity transform yields the 20 mm square") {
        const Polygon hull   = object->convex_hull_2d(Geometry::Transformation{}.get_matrix());
        const BoundingBox bb = hull.bounding_box();
        CHECK(hull.size() == 4);
        CHECK(bb.min.x() == scaled(0.));
        CHECK(bb.min.y() == scaled(0.));
        CHECK(bb.max.x() == scaled(20.));
        CHECK(bb.max.y() == scaled(20.));
    }

    SECTION("scaling and offset move and grow the footprint") {
        Geometry::Transformation t;
        t.set_scaling_factor({2, 2, 2}); // cube now spans [0,40]
        t.set_offset({10, 5, 0});        // then shift +10 in X, +5 in Y

        const Polygon hull   = object->convex_hull_2d(t.get_matrix());
        const BoundingBox bb = hull.bounding_box();
        CHECK(hull.size() == 4);
        CHECK(bb.min.x() == scaled(10.));
        CHECK(bb.min.y() == scaled(5.));
        CHECK(bb.max.x() == scaled(50.));
        CHECK(bb.max.y() == scaled(45.));
    }
}

TEST_CASE("An object's raw mesh keeps the triangles of each part on its own vertices", "[Model]")
{
    Model model;
    ModelObject *object = model.add_object();
    object->add_volume(make_cube(10, 10, 10), ModelVolumeType::MODEL_PART, false);
    TriangleMesh second = make_cube(10, 10, 10);
    second.translate(30, 0, 0);
    object->add_volume(std::move(second), ModelVolumeType::MODEL_PART, false);

    // Two separate cubes stay two closed components, one around each cube.
    const std::vector<indexed_triangle_set> parts = its_split(object->raw_indexed_triangle_set());
    REQUIRE(parts.size() == 2);
    std::vector<double> min_x;
    for (const indexed_triangle_set &part : parts) {
        CHECK(part.indices.size() == 12);
        min_x.push_back(bounding_box(part).min.x());
    }
    std::sort(min_x.begin(), min_x.end());
    CHECK_THAT(min_x.front(), Catch::Matchers::WithinAbs(0., 1e-4));
    CHECK_THAT(min_x.back(), Catch::Matchers::WithinAbs(30., 1e-4));
}

TEST_CASE("An instance added from another's scale, rotation and mirror matches it", "[Model]")
{
    // Add instance and Fill bed copy an instance this way.
    const int case_idx = GENERATE(0, 1);
    Transform3d trafo = Transform3d::Identity();
    if (case_idx == 0)
        trafo.linear() = Eigen::AngleAxisd(0.5 * PI, Vec3d::UnitZ()).toRotationMatrix() * Vec3d(-1., 1., 1.).asDiagonal();
    else
        trafo.linear() << 4.4408921e-16,   0.819152044,    0.573576436,
                          1.0,            -4.4408921e-16, -1.11022302e-16,
                         -5.55111512e-17, -0.573576436,    0.819152044;
    CAPTURE(case_idx);

    Model model;
    ModelObject *object = model.add_object();
    object->add_volume(make_cube(10, 20, 30));
    ModelInstance *original = object->add_instance();
    original->set_transformation(Geometry::Transformation(trafo));
    const ModelInstance *copy = object->add_instance(original->get_offset(), original->get_scaling_factor(),
                                                     original->get_rotation(), original->get_mirror());
    CHECK(copy->get_matrix().linear().isApprox(original->get_matrix().linear(), 1e-9));
}

namespace {

// A flat triangle first, then a cube for each offset, all in one mesh.
TriangleMesh flat_triangle_then_cubes(const std::vector<double> &cube_offsets)
{
    indexed_triangle_set its;
    its.vertices = {{-20.f, 0.f, 0.f}, {-10.f, 0.f, 0.f}, {-15.f, 5.f, 0.f}};
    its.indices  = {{0, 1, 2}};
    for (double offset : cube_offsets) {
        TriangleMesh cube = make_cube(10, 10, 10);
        cube.translate(float(offset), 0.f, 0.f);
        its_merge(its, cube.its);
    }
    return TriangleMesh(its);
}

} // namespace

TEST_CASE("Splitting a part drops pieces enclosing no volume and keeps the part itself", "[Model]")
{
    Model model;
    ModelObject *object = model.add_object();
    ModelVolume *volume = object->add_volume(flat_triangle_then_cubes({0., 30.}), ModelVolumeType::MODEL_PART, false);
    REQUIRE(volume->is_splittable());

    CHECK(volume->split(1, false) == 2);
    REQUIRE(object->volumes.size() == 2);
    CHECK(std::find(object->volumes.begin(), object->volumes.end(), volume) != object->volumes.end());
    for (const ModelVolume *part : object->volumes) {
        CHECK(part->mesh().facets_count() == 12);
        CHECK_FALSE(part->is_splittable());
    }
}

TEST_CASE("Splitting a part with a single solid piece keeps just that piece", "[Model]")
{
    Model model;
    ModelObject *object = model.add_object();
    ModelVolume *volume = object->add_volume(flat_triangle_then_cubes({0.}), ModelVolumeType::MODEL_PART, false);

    CHECK(volume->split(1, false) == 1);
    REQUIRE(object->volumes.size() == 1);
    CHECK(object->volumes.front() == volume);
    CHECK(volume->mesh().facets_count() == 12);
}

TEST_CASE("Splitting a part made only of pieces enclosing no volume leaves it whole", "[Model]")
{
    Model model;
    ModelObject *object = model.add_object();
    indexed_triangle_set its;
    its.vertices = {{0.f, 0.f, 0.f}, {10.f, 0.f, 0.f}, {5.f, 5.f, 0.f}, {20.f, 0.f, 0.f}, {30.f, 0.f, 0.f}, {25.f, 5.f, 0.f}};
    its.indices  = {{0, 1, 2}, {3, 4, 5}};
    ModelVolume *volume = object->add_volume(TriangleMesh(its), ModelVolumeType::MODEL_PART, false);

    CHECK(volume->split(1, false) == 1);
    REQUIRE(object->volumes.size() == 1);
    CHECK(object->volumes.front() == volume);
    CHECK(volume->mesh().facets_count() == 2);
}

TEST_CASE("Replacing a part's mesh updates whether it can be split", "[Model]")
{
    Model model;
    ModelObject *object = model.add_object();
    TriangleMesh two_cubes = make_cube(10, 10, 10);
    TriangleMesh second    = make_cube(10, 10, 10);
    second.translate(30.f, 0.f, 0.f);
    two_cubes.merge(second);
    ModelVolume *volume = object->add_volume(std::move(two_cubes), ModelVolumeType::MODEL_PART, false);
    REQUIRE(volume->is_splittable());

    volume->set_mesh(make_cube(10, 10, 10));
    CHECK_FALSE(volume->is_splittable());
}

TEST_CASE("Repairing a part replaces its mesh only when the repair changes it", "[Model]")
{
    Model model;
    ModelObject *object = model.add_object();
    TriangleMesh flipped = make_cube(10, 10, 10);
    std::swap(flipped.its.indices.front()[1], flipped.its.indices.front()[2]);
    ModelVolume *broken = object->add_volume(std::move(flipped), ModelVolumeType::MODEL_PART, false);
    TriangleMesh cube = make_cube(10, 10, 10);
    cube.translate(30.f, 0.f, 0.f);
    ModelVolume *valid = object->add_volume(std::move(cube), ModelVolumeType::MODEL_PART, false);
    const ObjectID broken_id = broken->id();
    const ObjectID valid_id  = valid->id();

    MeshRepairStats stats;
    REQUIRE(broken->repair(false, &stats));
    CHECK(stats.modified);
    CHECK(broken->id() != broken_id);
    CHECK_THAT(its_volume(broken->mesh().its), Catch::Matchers::WithinRel(1000., 1e-6));
    CHECK_FALSE(broken->get_convex_hull().empty());

    REQUIRE(valid->repair(false, &stats));
    CHECK_FALSE(stats.modified);
    CHECK(valid->id() == valid_id);
}

namespace {

void check_near(const Vec3d &actual, const Vec3d &expected)
{
    INFO("actual " << actual.transpose() << ", expected " << expected.transpose());
    for (int i = 0; i < 3; ++i)
        CHECK_THAT(actual[i], Catch::Matchers::WithinAbs(expected[i], 1e-6));
}

// The centre of a part's bounding box in object coordinates.
Vec3d part_center(const ModelVolume &volume) { return volume.mesh().transformed_bounding_box(volume.get_matrix()).center(); }

Vec3d bottom_center(const BoundingBoxf3 &bb) { return Vec3d(bb.center().x(), bb.center().y(), bb.min.z()); }

// A cube of the given size whose minimum corner is at `min`.
TriangleMesh cube_at(double size, const Vec3d &min)
{
    TriangleMesh mesh = make_cube(size, size, size);
    mesh.translate(min.cast<float>());
    return mesh;
}

} // namespace

TEST_CASE("Scaling a multi-part object scales where its parts sit, not just their size", "[Model][Regression]")
{
    // A 20 mm body around the origin and a 4 mm inlay centred at (8, 6, -4), the way an imported
    // multi-material object is laid out: each part's mesh centred on itself and placed by its own offset.
    Model        model;
    ModelObject *object = model.add_object();
    object->add_volume(cube_at(20., Vec3d(-10., -10., -10.)));
    ModelVolume *inlay = object->add_volume(cube_at(4., Vec3d(6., 4., -6.)));
    REQUIRE_FALSE(inlay->get_offset().isZero());
    check_near(part_center(*inlay), Vec3d(8., 6., -4.));

    SECTION("uniform scale")
    {
        object->scale(0.5);
        check_near(part_center(*object->volumes[0]), Vec3d::Zero());
        check_near(part_center(*inlay), Vec3d(4., 3., -2.));
        check_near(inlay->mesh().transformed_bounding_box(inlay->get_matrix()).size(), Vec3d(2., 2., 2.));
    }
    SECTION("a different scale on each axis")
    {
        object->scale(0.5, 1., 2.);
        check_near(part_center(*inlay), Vec3d(4., 6., -8.));
        check_near(inlay->mesh().transformed_bounding_box(inlay->get_matrix()).size(), Vec3d(2., 4., 8.));
    }
}

TEST_CASE("Scaling in place keeps each instance's place on the plate and the height of its lowest point", "[Model]")
{
    // The command line loads a mesh in its file coordinates: the part is centred on itself and placed by its
    // offset, so the object's origin can be far from the object, and scale() alone would move it.
    Model        model;
    ModelObject *object = model.add_object();
    object->add_volume(cube_at(10., Vec3d(100., 50., 5.)));
    object->add_instance();

    SECTION("an object resting on the bed stays on it")
    {
        object->ensure_on_bed();
        REQUIRE_THAT(object->instance_bounding_box(0).min.z(), Catch::Matchers::WithinAbs(0., 1e-6));
        object->scale_in_place(0.5);
        check_near(bottom_center(object->instance_bounding_box(0)), Vec3d(105., 55., 0.));
        check_near(object->instance_bounding_box(0).size(), Vec3d(5., 5., 5.));
    }
    SECTION("an object above the bed keeps its height")
    {
        object->scale_in_place(2.);
        check_near(bottom_center(object->instance_bounding_box(0)), Vec3d(105., 55., 5.));
        check_near(object->instance_bounding_box(0).size(), Vec3d(20., 20., 20.));
    }
}

TEST_CASE("Scaling a multi-part object in place keeps its parts together and every instance in its place", "[Model]")
{
    // A 20 mm body with a 4 mm inlay sunk into its top face, away from the origin, placed twice: once as is and
    // once turned a quarter turn about Z at another spot on the plate.
    Model        model;
    ModelObject *object = model.add_object();
    object->add_volume(cube_at(20., Vec3d(40., 40., 0.)));
    ModelVolume *inlay = object->add_volume(cube_at(4., Vec3d(52., 46., 16.)));
    object->add_instance();
    ModelInstance *turned = object->add_instance();
    turned->set_rotation(Vec3d(0., 0., PI / 2.));
    turned->set_offset(Vec3d(-50., 150., 0.));

    std::vector<Vec3d> places;
    for (size_t i = 0; i < object->instances.size(); ++i)
        places.emplace_back(bottom_center(object->instance_bounding_box(i)));
    const Vec3d inlay_from_body = part_center(*inlay) - part_center(*object->volumes[0]);

    object->scale_in_place(0.5);

    for (size_t i = 0; i < object->instances.size(); ++i) {
        INFO("instance " << i);
        check_near(bottom_center(object->instance_bounding_box(i)), places[i]);
        check_near(object->instance_bounding_box(i).size(), Vec3d(10., 10., 10.));
    }
    check_near(part_center(*inlay) - part_center(*object->volumes[0]), 0.5 * inlay_from_body);
}
