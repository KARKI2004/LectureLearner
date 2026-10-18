# ============================================================
#  Lecture Learner
#  Team Pythons · CMPS 4010 · Survey of Programming Languages
#
#      recording ──▶ (1) TRANSCRIBE ──▶ transcript ──▶ (2) NOTES ──▶ notes.md
#
#  Cloud (default)
#    (1) OpenAI Whisper API           audio -> text, ~10 s for a short clip
#    (2) Claude API  or  OpenAI API   text  -> main points, sub-points, announcements
#
#  Pure-Julia fallback (TextTools.jl) — kicks in automatically if a
#  key is missing or a request fails, or always with --offline
#    (1) none: you must supply a transcript (--text)
#    (2) TF-IDF + TextTiling summary, regex date finder
#
#  Usage
#    julia LectureLearner.jl lecture.m4a            full pipeline (any audio format)
#    julia LectureLearner.jl --text transcript.txt  skip transcription
#    julia LectureLearner.jl --offline --text transcript.txt
#
#  Keys (set as environment variables — never put them in this file)
#    OPENAI_API_KEY      needed for transcription; also notes if NOTES_BACKEND = :openai
#    ANTHROPIC_API_KEY   needed for notes if NOTES_BACKEND = :claude
# ============================================================

include("TextTools.jl")
using .TextTools

# Keep the transcript-only path usable without cloud packages installed.
function load_cloud_packages()
    try
        @eval import HTTP, JSON3
    catch e
        error("Cloud mode requires HTTP and JSON3. Run `julia --project=. -e 'using Pkg; Pkg.instantiate()'` first. Original error: $(sprint(showerror, e))")
    end
end

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
const NOTES_BACKEND = :claude                # :claude or :openai
const CLAUDE_MODEL  = "claude-haiku-5-5"     # fast + cheap; "claude-sonnet-5-5" for better notes
const OPENAI_MODEL  = "gpt-4o-mini"
const WHISPER_MODEL = "whisper-1"

openai_key()    = get(ENV, "OPENAI_API_KEY", "")
anthropic_key() = get(ENV, "ANTHROPIC_API_KEY", "")

# ------------------------------------------------------------
# Stage 1 — Transcription (OpenAI Whisper API)
# ------------------------------------------------------------
"""
Upload an audio file (mp3, m4a, wav, webm, mp4 … up to 25 MB) and return the transcript.
"""
function transcribe_audio(path::AbstractString)
    isempty(openai_key()) && error("OPENAI_API_KEY is not set — transcription needs it")
    load_cloud_packages()
    @info "Uploading audio for transcription" file = path
    return open(path) do file
        form = HTTP.Form(Dict("file" => file, "model" => WHISPER_MODEL, "response_format" => "text"))
        r = HTTP.post("https://api.openai.com/v1/audio/transcriptions",
                      ["Authorization" => "Bearer " * openai_key()], form; readtimeout = 300)
        String(r.body)
    end
end

# ------------------------------------------------------------
# Stage 2 — Notes via LLM
# ------------------------------------------------------------
const NOTES_PROMPT = """
You are a study assistant. Below is a transcript of a college lecture.
Produce study notes as JSON with exactly this shape:

{
  "title": "short lecture title",
  "main_points": [
    {"point": "one clear sentence", "sub_points": ["detail", "detail"]}
  ],
  "announcements": [
    {"type": "test | deadline | schedule_change | upcoming_class",
     "detail": "what was said, cleaned up",
     "date": "date or day as stated, or null"}
  ]
}

Rules:
- Rewrite spoken language as concise note-style sentences.
- Group related ideas; one main point per topic the professor covered, in lecture order.
- Put every exam, quiz, homework due date, cancelled class, or "read X before next class" in announcements.
- If the lecture has no announcements, return an empty list.
- Output only the JSON, no commentary, no code fences.

TRANSCRIPT:
"""

function ask_claude(prompt)
    isempty(anthropic_key()) && error("ANTHROPIC_API_KEY is not set")
    body = JSON3.write(Dict(
        "model" => CLAUDE_MODEL, "max_tokens" => 4000,
        "messages" => [Dict("role" => "user", "content" => prompt)]))
    r = HTTP.post("https://api.anthropic.com/v1/messages",
                  ["x-api-key" => anthropic_key(), "anthropic-version" => "2023-06-01",
                   "Content-Type" => "application/json"], body; readtimeout = 120)
    return JSON3.read(r.body).content[1].text
end

function ask_openai(prompt)
    isempty(openai_key()) && error("OPENAI_API_KEY is not set")
    body = JSON3.write(Dict(
        "model" => OPENAI_MODEL,
        "response_format" => Dict("type" => "json_object"),
        "messages" => [Dict("role" => "user", "content" => prompt)]))
    r = HTTP.post("https://api.openai.com/v1/chat/completions",
                  ["Authorization" => "Bearer " * openai_key(), "Content-Type" => "application/json"],
                  body; readtimeout = 120)
    return JSON3.read(r.body).choices[1].message.content
end

"Ask the LLM for notes. Returns a parsed JSON object with title / main_points / announcements."
function llm_notes(transcript::AbstractString)
    load_cloud_packages()
    prompt = NOTES_PROMPT * transcript
    raw = NOTES_BACKEND == :claude ? ask_claude(prompt) : ask_openai(prompt)
    raw = strip(replace(raw, r"^```(?:json)?\s*|\s*```$" => ""))   # strip fences if any
    return JSON3.read(raw)
end

"Check the cloud response before rendering, while fallback is still possible."
function validate_notes(notes)
    hasproperty(notes, :title) && notes.title isa AbstractString && !isempty(strip(notes.title)) || error("Cloud notes have no valid title")
    hasproperty(notes, :main_points) && notes.main_points isa AbstractVector || error("Cloud notes have no main_points array")
    hasproperty(notes, :announcements) && notes.announcements isa AbstractVector || error("Cloud notes have no announcements array")
    for point in notes.main_points
        hasproperty(point, :point) && point.point isa AbstractString && !isempty(strip(point.point)) || error("Cloud notes contain an invalid main point")
        hasproperty(point, :sub_points) && point.sub_points isa AbstractVector && all(x -> x isa AbstractString, point.sub_points) || error("Cloud notes contain invalid sub_points")
    end
    for announcement in notes.announcements
        hasproperty(announcement, :type) && announcement.type isa AbstractString && haskey(TextTools.ICON, String(announcement.type)) || error("Cloud notes contain an invalid announcement type")
        hasproperty(announcement, :detail) && announcement.detail isa AbstractString && !isempty(strip(announcement.detail)) || error("Cloud notes contain an invalid announcement detail")
        hasproperty(announcement, :date) && (announcement.date === nothing || announcement.date isa AbstractString) || error("Cloud notes contain an invalid announcement date")
    end
    return notes
end

"""
Cloud first, pure-Julia second. Returns (notes, mode) where mode is "cloud" or "offline".
"""
function make_notes(transcript::AbstractString; offline::Bool = false, cloud_notes = llm_notes)
    if !offline
        try
            return validate_notes(cloud_notes(transcript)), "cloud"
        catch e
            @warn "Cloud notes failed — falling back to pure-Julia summarizer" error_type = string(typeof(e))
        end
    end
    return TextTools.analyze(transcript), "offline"
end

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
function main(args)
    offline = count(==("--offline"), args) == 1
    count(==("--offline"), args) <= 1 || throw(ArgumentError("--offline may be specified only once"))
    inputs = filter(!=("--offline"), args)
    usage = "Usage: julia --project=. LectureLearner.jl [--offline] --text transcript.txt\n       julia --project=. LectureLearner.jl lecture.m4a"
    length(inputs) in (1, 2) || throw(ArgumentError(usage))
    is_text = length(inputs) == 2 && inputs[1] == "--text"
    is_audio = length(inputs) == 1 && !startswith(inputs[1], "-")
    (is_text || is_audio) || throw(ArgumentError(usage))
    offline && is_audio && throw(ArgumentError("Offline mode cannot transcribe audio. Use --text transcript.txt"))
    path = inputs[end]
    isfile(path) || throw(ArgumentError("Input file not found: $path"))
    base = splitext(path)[1]
    if is_text
        transcript = read(path, String)
    else
        transcript = transcribe_audio(path)
        isempty(strip(transcript)) && throw(ArgumentError("Transcription returned empty text"))
        write(base * "_transcript.txt", transcript)
        @info "Transcript saved" file = base * "_transcript.txt"
    end
    isempty(strip(transcript)) && throw(ArgumentError("Input transcript is empty: $path"))

    t0 = time()
    notes, mode = make_notes(transcript; offline = offline)
    t = time() - t0
    md = TextTools.render(notes)
    out = base * "_notes.md"
    write(out, md)
    println(md)
    @info "Done" mode seconds = round(t; digits = 1) saved = out
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        main(ARGS)
    catch e
        println(stderr, "Error: ", sprint(showerror, e))
        exit(1)
    end
end
