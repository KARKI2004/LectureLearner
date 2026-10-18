"""Local recording-to-text transcription using FFmpeg and whisper.cpp."""
module LocalTranscription

export transcribe, main

const USAGE = """
Usage: julia --project=. transcribe_local.jl RECORDING --model MODEL.bin [options]

Options:
  --output PATH      Transcript destination (default: transcripts/<name>_transcript.txt)
  --start SECONDS    Start at this time in the recording (default: 0)
  --duration SECONDS Transcribe only this many seconds
  --help             Show this help

The recording and model stay local. Existing transcript files are not overwritten.
"""

function seconds(value::String, flag::String; allow_zero::Bool)
    parsed = tryparse(Float64, value)
    (parsed !== nothing && isfinite(parsed) && (allow_zero ? parsed >= 0 : parsed > 0)) ||
        throw(ArgumentError("$flag needs a $(allow_zero ? "nonnegative" : "positive") number of seconds"))
    return parsed
end

function parse_args(args)
    length(args) == 1 && args[1] == "--help" && return nothing
    recording = nothing
    model = nothing
    output = nothing
    start = 0.0
    duration = nothing
    seen = Set{String}()
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("--model", "--output", "--start", "--duration")
            arg in seen && throw(ArgumentError("$arg may be supplied only once\n$USAGE"))
            push!(seen, arg)
            i < length(args) || throw(ArgumentError("Missing value for $arg\n$USAGE"))
            value = args[i + 1]
            (isempty(value) || startswith(value, "--")) &&
                throw(ArgumentError("Missing value for $arg\n$USAGE"))
            if arg == "--model"
                model = value
            elseif arg == "--output"
                output = value
            elseif arg == "--start"
                start = seconds(value, arg; allow_zero = true)
            else
                duration = seconds(value, arg; allow_zero = false)
            end
            i += 2
        elseif startswith(arg, "-")
            throw(ArgumentError("Unknown option: $arg\n$USAGE"))
        elseif recording === nothing
            recording = arg
            i += 1
        else
            throw(ArgumentError("Only one recording may be supplied\n$USAGE"))
        end
    end
    recording === nothing && throw(ArgumentError("Recording path is required\n$USAGE"))
    model === nothing && throw(ArgumentError("--model is required\n$USAGE"))
    return (recording = recording, model = model, output = output,
            start = start, duration = duration)
end

function executable(name::String, install_hint::String)
    path = Sys.which(name)
    path === nothing && throw(ArgumentError("$name was not found on PATH. $install_hint"))
    return path
end

function run_checked(args::Vector{String}, label::String)
    mktemp() do _, log
        process = run(pipeline(ignorestatus(Cmd(args)), stdout = devnull, stderr = log))
        if !success(process)
            flush(log)
            seekstart(log)
            detail = strip(read(log, String))
            detail = isempty(detail) ? "no error detail was reported" : last(detail, min(length(detail), 700))
            throw(ErrorException("$label failed: $detail"))
        end
    end
end

function plain_text(raw::AbstractString)
    without_times = replace(raw,
        r"(?m)^[ \t]*\[\d{2}:\d{2}:\d{2}\.\d+[ \t]+-->[ \t]+\d{2}:\d{2}:\d{2}\.\d+\][ \t]*" => "")
    lines = [strip(line) for line in split(without_times, '\n') if !isempty(strip(line))]
    return isempty(lines) ? "" : join(lines, "\n") * "\n"
end

"""Extract 16 kHz mono WAV, transcribe locally, and save plain text."""
function transcribe(recording::AbstractString, model::AbstractString;
                    output::Union{Nothing,AbstractString} = nothing,
                    start::Real = 0, duration::Union{Nothing,Real} = nothing,
                    ffmpeg::Union{Nothing,AbstractString} = nothing,
                    whisper::Union{Nothing,AbstractString} = nothing)
    isfile(recording) || throw(ArgumentError("Recording file not found: $recording"))
    filesize(recording) > 0 || throw(ArgumentError("Recording file is empty: $recording"))
    isfile(model) || throw(ArgumentError("Model file not found: $model"))
    filesize(model) > 0 || throw(ArgumentError("Model file is empty: $model"))
    isfinite(start) && start >= 0 || throw(ArgumentError("--start must be nonnegative"))
    (duration === nothing || (isfinite(duration) && duration > 0)) ||
        throw(ArgumentError("--duration must be positive"))

    base = splitext(basename(recording))[1]
    destination = output === nothing ? joinpath(@__DIR__, "transcripts", base * "_transcript.txt") : abspath(output)
    ispath(destination) && throw(ArgumentError("Transcript already exists: $destination. Choose --output to keep both runs."))
    ffmpeg_path = ffmpeg === nothing ? executable("ffmpeg", "Install it with `brew install ffmpeg`.") : executable(String(ffmpeg), "Check the FFmpeg path.")
    whisper_path = whisper === nothing ? executable("whisper-cli", "Install it with `brew install whisper.cpp`.") : executable(String(whisper), "Check the whisper-cli path.")

    started = time()
    transcript = mktempdir() do dir
        wav = joinpath(dir, "audio.wav")
        prefix = joinpath(dir, "whisper-output")
        ffargs = String[ffmpeg_path, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
        start > 0 && append!(ffargs, ["-ss", string(start)])
        append!(ffargs, ["-i", abspath(recording)])
        duration === nothing || append!(ffargs, ["-t", string(duration)])
        append!(ffargs, ["-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
                         "-c:a", "pcm_s16le", "-f", "wav", wav])
        run_checked(ffargs, "FFmpeg audio extraction")
        isfile(wav) && filesize(wav) > 44 || error("FFmpeg produced no usable audio. Check that the recording has an audio track.")

        run_checked(String[whisper_path, "-m", abspath(model), "-f", wav,
                           "-otxt", "-of", prefix, "-nt", "-np"], "whisper.cpp transcription")
        generated = prefix * ".txt"
        isfile(generated) || error("whisper.cpp did not write a transcript file")
        text = plain_text(read(generated, String))
        isempty(text) && error("whisper.cpp returned an empty transcript")
        text
    end

    mkpath(dirname(destination))
    ispath(destination) && throw(ArgumentError("Transcript already exists: $destination"))
    write(destination, transcript)
    return (path = destination, elapsed_seconds = time() - started)
end

function main(args)
    options = parse_args(args)
    if options === nothing
        print(USAGE)
        return
    end
    result = transcribe(options.recording, options.model;
                        output = options.output, start = options.start,
                        duration = options.duration)
    println("Transcript saved: ", result.path)
    println("Elapsed processing time: ", round(result.elapsed_seconds; digits = 1), " seconds")
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    try
        LocalTranscription.main(ARGS)
    catch e
        println(stderr, "Error: ", sprint(showerror, e))
        exit(1)
    end
end
