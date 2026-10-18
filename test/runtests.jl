using Test
using JSON3

include(joinpath(@__DIR__, "..", "LectureLearner.jl"))

const SAMPLE = read(joinpath(@__DIR__, "..", "sample_transcript.txt"), String)

@testset "Offline notes preserve lecture content" begin
    notes, mode = make_notes(SAMPLE; offline = true)
    md = TextTools.render(notes)
    @test mode == "offline"
    @test length(notes.announcements) == 3
    @test occursin("midterm exam", md)
    @test occursin("Homework 4", md)
    @test occursin("Class is cancelled", md)
    @test occursin("new test data", md)
    @test occursin("Backpropagation", md)
    @test !occursin("[TEST] An overfit model", md)
    @test !occursin("That's all for today", md)
    for topic in ("unit test verifies gradients", "final project uses a validation set", "lab exercise explains loss functions", "Project 2 uses a validation set")
        result = TextTools.analyze("The $topic.")
        @test isempty(result.announcements)
        @test length(result.main_points) == 1
    end
    @test length(TextTools.analyze("The quiz is Friday.").announcements) == 1
    @test length(TextTools.analyze("Project 2 is due next Friday.").announcements) == 1
end

@testset "Cloud response validation and fallback" begin
    good = JSON3.read("""{"title":"Networks","main_points":[{"point":"Weights change","sub_points":["Gradient descent"]}],"announcements":[]}""")
    @test make_notes("Gradient descent changes weights."; cloud_notes = _ -> good)[2] == "cloud"
    malformed = (
        JSON3.read("""{"title":"Networks","main_points":[{"point":"Weights change"}],"announcements":[]}"""),
        JSON3.read("""{"title":"Networks","main_points":[],"announcements":[{"type":"test","detail":"Quiz","date":42}]}"""),
        JSON3.read("""{"title":"Networks","main_points":[],"announcements":null}"""),
    )
    for bad in malformed
        @test_throws Exception validate_notes(bad)
        notes, mode = make_notes("Gradient descent changes weights."; cloud_notes = _ -> bad)
        @test mode == "offline"
        @test occursin("Gradient descent", TextTools.render(notes))
    end
end

@testset "Input errors" begin
    mktempdir() do dir
        missing = joinpath(dir, "missing.txt")
        empty = joinpath(dir, "empty.txt")
        write(empty, " \n")
        @test_throws ArgumentError main(["--offline", "--text", missing])
        @test_throws ArgumentError main(["--offline", "--text", empty])
        @test_throws ArgumentError main(["--offline", "audio.m4a"])
        @test_throws ArgumentError main(["--text"])
        @test_throws ArgumentError main(["--bogus"])
        @test_throws ArgumentError main(["--offline", "--offline", "--text", empty])
        @test_throws ArgumentError main(["--text", empty, "extra"])
    end
end

@testset "Offline startup without cloud packages" begin
    mktempdir() do depot
        script = "include($(repr(joinpath(@__DIR__, "..", "LectureLearner.jl")))); notes, mode = make_notes(\"The unit test verifies gradients.\"; offline=true); @assert mode == \"offline\"; @assert isempty(notes.announcements)"
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(joinpath(@__DIR__, "..")) -e $script`
        @test success(setenv(cmd, "JULIA_DEPOT_PATH" => depot))
    end
end
