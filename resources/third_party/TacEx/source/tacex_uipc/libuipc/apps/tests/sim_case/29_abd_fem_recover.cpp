#include <catch2/catch_all.hpp>
#include <app/asset_dir.h>
#include <uipc/uipc.h>
#include <uipc/constitution/stable_neo_hookean.h>
#include <uipc/constitution/arap.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <uipc/core/affine_body_state_accessor_feature.h>
#include <filesystem>
#include <fstream>

static void clear()
{
    using namespace uipc;
    using namespace uipc::core;
    using namespace uipc::geometry;
    using namespace uipc::constitution;

    namespace fs = std::filesystem;

    std::string tetmesh_dir{AssetDir::tetmesh_path()};
    auto        this_output_path = AssetDir::output_path(__FILE__);
    auto        dump_path        = fmt::format("{}/dump/", this_output_path);
    auto        count            = fs::remove_all(dump_path);
    Logger::current_logger().info("Remove {} entries in {}", count, dump_path);
}

static void run(int I, bool friction = false)
{
    using namespace uipc;
    using namespace uipc::core;
    using namespace uipc::geometry;
    using namespace uipc::constitution;

    namespace fs = std::filesystem;

    // Test Config:

    constexpr SizeT EndFrame     = 50;
    constexpr SizeT RecoverFrame = 40;


    std::string tetmesh_dir{AssetDir::tetmesh_path()};
    auto        this_output_path = AssetDir::output_path(__FILE__);


    Engine engine{"cuda", this_output_path};
    World  world{engine};

    auto config = Scene::default_config();

    config["gravity"]                       = Vector3{0, -9.8, 0};
    config["contact"]["enable"]             = true;
    config["contact"]["friction"]["enable"] = friction;
    config["line_search"]["max_iter"]       = 8;
    config["linear_system"]["tol_rate"]     = 1e-3;
    config["line_search"]["report_energy"]  = false;
    if(I == 2)  // recover at specified frame when running the third time
    {
        config["recovery"]["frame"] = RecoverFrame;
    }

    {  // dump config
        std::ofstream ofs(fmt::format("{}config.json", this_output_path));
        ofs << config.dump(4);
    }

    SimplicialComplexIO io;

    Scene scene{config};
    {
        // create constitution and contact model
        StableNeoHookean       snh;
        AffineBodyConstitution abd;

        scene.contact_tabular().default_model(0.5, 1.0_GPa);
        auto default_element = scene.contact_tabular().default_element();

        // create object
        auto object = scene.objects().create("tets");

        auto lower_mesh = io.read(fmt::format("{}cube.msh", tetmesh_dir));

        label_surface(lower_mesh);
        label_triangle_orient(lower_mesh);

        SimplicialComplex upper_mesh = lower_mesh;

        auto parm = ElasticModuli::youngs_poisson(20.0_kPa, 0.49);
        snh.apply_to(lower_mesh, parm);
        abd.apply_to(upper_mesh, 1.0_MPa);

        constexpr SizeT N = 6;

        for(SizeT i = 0; i < N; i++)
        {
            if(i % 2 == 0)
            {
                SimplicialComplex l = lower_mesh;
                {
                    auto pos_v = view(l.positions());
                    for(auto& p : pos_v)
                        p.y() += 1.2 * i;
                }
                object->geometries().create(l);
            }
            else
            {
                SimplicialComplex u = upper_mesh;
                {
                    auto pos_v = view(u.positions());
                    for(auto& p : pos_v)
                        p.y() += 1.2 * i;
                }
                object->geometries().create(u);
            }
        }

        auto g = ground(-0.6);
        object->geometries().create(g);
    }

    world.init(scene);
    REQUIRE(world.is_valid());
    SceneIO sio{scene};
    sio.write_surface(
        fmt::format("{}scene_surface{}.obj", this_output_path, world.frame()));


    if(I == 0)
    {
        // MUST fail at the first run
        REQUIRE(!world.recover());
    }
    else if(I == 1)
    {
        // MUST recover at the max frame
        REQUIRE(world.recover());
        REQUIRE(world.frame() == EndFrame);
    }
    else if(I == 2)
    {
        auto specified_frame = 40;
        // MUST recover at the specified frame
        REQUIRE(world.recover(specified_frame));
        REQUIRE(world.frame() == specified_frame);
    }

    while(world.frame() < EndFrame)
    {
        world.advance();
        world.retrieve();
        if(I == 0)  // only dump in the first run
            world.dump();
        sio.write_surface(
            fmt::format("{}scene_surface{}.obj", this_output_path, world.frame()));
    }

    if(friction && I == 0)
    {
        // Rewind the same world after contact has developed. The saved frame
        // has different contacts, so the previous trajectory-filter cache must
        // not be used as its lagged friction set.
        for(int episode = 0; episode < 3; ++episode)
        {
            REQUIRE(world.recover(1));
            world.retrieve();
            while(world.frame() < EndFrame)
            {
                world.advance();
                world.retrieve();
                REQUIRE(world.is_valid());
                auto surface = sio.simplicial_surface(2);
                for(const auto& position : surface.positions().view())
                    REQUIRE(position.allFinite());
            }
        }
    }
}


TEST_CASE("29_abd_fem_recover", "[abd_fem]")
{
    clear();
    // 0: dump the infos
    // 1: recover at the max frame
    // 2: recover at the specified frame
    for(int I = 0; I < 3; I++)
    {
        run(I);
    }
}

TEST_CASE("29_abd_fem_recover_with_friction", "[abd_fem][friction][recover]")
{
    clear();
    run(0, true);
}

TEST_CASE("29_recover_parallel_edges_with_stale_friction", "[abd][friction][recover]")
{
    using namespace uipc;
    using namespace uipc::core;
    using namespace uipc::geometry;
    using namespace uipc::constitution;
    clear();
    Engine engine{"cuda", AssetDir::output_path(__FILE__)};
    World world{engine};
    auto config = Scene::default_config();
    config["gravity"] = Vector3{0, 0, 0};
    config["contact"]["friction"]["enable"] = true;
    config["contact"]["d_hat"] = 0.01;
    Scene scene{config};
    AffineBodyConstitution abd;
    scene.contact_tabular().default_model(2.0, 1.0_MPa);
    auto object = scene.objects().create("edge_contact");
    for(int body = 0; body < 2; ++body)
    {
        Float side = body == 0 ? -1.0 : 1.0;
        vector<Vector3> vertices = {{-0.3, 0, 0}, {0.3, 0, 0},
                                    {0, side * 0.3, 0.3}, {0, side * 0.3, -0.3}};
        vector<Vector4i> tets = {body == 0 ? Vector4i{0, 1, 2, 3} : Vector4i{0, 1, 3, 2}};
        auto mesh = tetmesh(vertices, tets);
        label_surface(mesh);
        label_triangle_orient(mesh);
        abd.apply_to(mesh, 100.0_MPa);
        scene.contact_tabular().default_element().apply_to(mesh);
        view(*mesh.instances().find<IndexT>(builtin::is_fixed))[0] = body == 0;
        Transform pose = Transform::Identity();
        pose.translation().y() = body == 0 ? 0.0 : 0.1;
        view(mesh.transforms())[0] = pose.matrix();
        object->geometries().create(mesh);
    }
    world.init(scene);
    REQUIRE(world.is_valid());
    world.advance();
    world.retrieve();
    world.dump();
    auto accessor = world.features().find<AffineBodyStateAccessorFeature>();
    REQUIRE(accessor);
    auto state = accessor->create_geometry(1, 1);
    state.instances().create<Matrix4x4>(builtin::transform);
    state.instances().create<Matrix4x4>(builtin::velocity);
    for(int episode = 0; episode < 3; ++episode)
    {
        accessor->copy_to(state);
        Transform pose = Transform::Identity();
        pose.rotate(Eigen::AngleAxis<Float>(0.7, Vector3::UnitY()));
        pose.translation().y() = 0.0005;
        view(state.transforms())[0] = pose.matrix();
        view(*state.instances().find<Matrix4x4>(builtin::velocity))[0].setZero();
        accessor->copy_from(state);
        world.advance();
        world.retrieve();
        // The current skew edges become parallel again in the saved frame.
        // Its old EE friction pair has no valid normal after recovery.
        REQUIRE(world.recover(1));
        world.retrieve();
        world.advance();
        world.retrieve();
        REQUIRE(world.is_valid());
        accessor->copy_to(state);
        REQUIRE(state.transforms().view()[0].allFinite());
    }
}
