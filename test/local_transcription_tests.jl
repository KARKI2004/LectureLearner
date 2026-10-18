include(joinpath(@__DIR__, "..", "transcribe_local.jl"))

@testset "Local transcription arguments" begin
    @test LocalTranscription.parse_args(["video.mp4", "--model", "models/model.bin", "--start", "4", "--duration", "180"]).duration == 180.0
    @test_throws ArgumentError LocalTranscription.parse_args(["video.mp4"])
    @test_throws ArgumentError LocalTranscription.parse_args(["video.mp4", "--model"])
    @test_throws ArgumentError LocalTranscription.parse_args(["video.mp4", "--model", "--output", "x"])
    @test_throws ArgumentError LocalTranscription.parse_args(["video.mp4", "--model", "x", "--duration", "0"])
    @test_throws ArgumentError LocalTranscription.parse_args(["video.mp4", "--model", "x", "--start", "-2"])
    @test_throws ArgumentError LocalTranscription.parse_args(["video.mp4", "--model", "x", "--bogus"])
end

@testset "Local subprocess paths, output, and cleanup" begin
    mktempdir() do dir
        recording = joinpath(dir, "lecture recording.mp4")
        model = joinpath(dir, "small model.bin")
        ffmpeg = joinpath(dir, "fake ffmpeg")
        whisper = joinpath(dir, "fake whisper-cli")
        ffmpeg_log = joinpath(dir, "ffmpeg args.log")
        whisper_log = joinpath(dir, "whisper args.log")
        output = joinpath(dir, "transcript output.txt")
        write(recording, "recording fixture")
        write(model, "model fixture")
        write(ffmpeg, raw"""#!/bin/sh
printf '%s\n' "$@" > "$LOCAL_FFMPEG_LOG"
if [ "$LOCAL_FFMPEG_FAIL" = 1 ]; then echo 'mock extraction error' >&2; exit 3; fi
for item do final="$item"; done
printf 'RIFF00000000000000000000000000000000000000000000000000' > "$final"
""")
        write(whisper, raw"""#!/bin/sh
printf '%s\n' "$@" > "$LOCAL_WHISPER_LOG"
if [ "$LOCAL_WHISPER_FAIL" = 1 ]; then echo 'mock model error' >&2; exit 4; fi
while [ "$#" -gt 0 ]; do
  if [ "$1" = '-of' ]; then shift; prefix="$1"; fi
  shift
done
printf '[00:00:00.000 --> 00:00:01.000] First sentence.\nSecond sentence.\n' > "$prefix.txt"
""")
        chmod(ffmpeg, 0o755)
        chmod(whisper, 0o755)

        withenv("LOCAL_FFMPEG_LOG" => ffmpeg_log, "LOCAL_WHISPER_LOG" => whisper_log) do
            result = LocalTranscription.transcribe(recording, model;
                output = output, start = 10, duration = 180, ffmpeg = ffmpeg, whisper = whisper)
            @test result.path == output
            @test result.elapsed_seconds >= 0
            @test read(output, String) == "First sentence.\nSecond sentence.\n"
            ffargs = split(chomp(read(ffmpeg_log, String)), '\n')
            whisper_args = split(chomp(read(whisper_log, String)), '\n')
            @test abspath(recording) in ffargs
            @test abspath(model) in whisper_args
            @test ["-ss", "10"] == ffargs[findfirst(==("-ss"), ffargs):findfirst(==("-ss"), ffargs)+1]
            @test ["-t", "180"] == ffargs[findfirst(==("-t"), ffargs):findfirst(==("-t"), ffargs)+1]
            @test !isfile(ffargs[end])
            @test_throws ArgumentError LocalTranscription.transcribe(recording, model;
                output = output, ffmpeg = ffmpeg, whisper = whisper)
        end

        failed_output = joinpath(dir, "failed.txt")
        withenv("LOCAL_FFMPEG_LOG" => ffmpeg_log, "LOCAL_FFMPEG_FAIL" => "1") do
            error_text = try
                LocalTranscription.transcribe(recording, model;
                    output = failed_output, ffmpeg = ffmpeg, whisper = whisper)
                ""
            catch e
                sprint(showerror, e)
            end
            @test occursin("FFmpeg audio extraction failed", error_text)
            @test occursin("mock extraction error", error_text)
            @test !isfile(failed_output)
            @test !isfile(last(split(chomp(read(ffmpeg_log, String)), '\n')))
        end
        withenv("LOCAL_FFMPEG_LOG" => ffmpeg_log, "LOCAL_WHISPER_LOG" => whisper_log,
                "LOCAL_WHISPER_FAIL" => "1") do
            error_text = try
                LocalTranscription.transcribe(recording, model;
                    output = failed_output, ffmpeg = ffmpeg, whisper = whisper)
                ""
            catch e
                sprint(showerror, e)
            end
            @test occursin("whisper.cpp transcription failed", error_text)
            @test occursin("mock model error", error_text)
            @test !isfile(failed_output)
            wav = split(chomp(read(whisper_log, String)), '\n')[4]
            @test !isfile(wav)
        end
        @test_throws ArgumentError LocalTranscription.transcribe("missing.mp4", model;
            ffmpeg = ffmpeg, whisper = whisper)
        @test_throws ArgumentError LocalTranscription.transcribe(recording, "missing.bin";
            ffmpeg = ffmpeg, whisper = whisper)
    end
end
