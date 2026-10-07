using Trixi
using TrixiMaxwell
using Test

# CI jobs set `TRIXI_TEST` to select a subset of the tests.
const TRIXI_TEST = get(ENV, "TRIXI_TEST", "all")

@time @testset "TrixiMaxwell.jl tests" begin
    @time if TRIXI_TEST == "all" || TRIXI_TEST == "unit"
        include("test_unit.jl")
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "type"
        include("test_type.jl")
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "dgmulti_3d"
        include("test_dgmulti_3d.jl")
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "dgsem_3d"
        include("test_dgsem_3d.jl")
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "meshes"
        include("test_meshes.jl")
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "upstream"
        @testset "Namespace conflicts" begin
            for name in names(Trixi)
                @test !(name in names(TrixiMaxwell))
            end
        end
    end
end
