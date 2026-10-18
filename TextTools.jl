# ============================================================
#  TextTools.jl — pure-Julia fallback for Lecture Learner
#  Team Pythons · CMPS 4010
#
#  No packages, no internet, no AI model. Used when the cloud
#  services are unavailable, and as the baseline the LLM output
#  is compared against.
#
#    find_announcements(sentences)  -> regex search for tests, deadlines, schedule changes
#    summarize(sentences, skip)     -> TF-IDF extractive summary: main point + sub-points per topic
#    render(text)                   -> full Markdown notes
# ============================================================
module TextTools

export split_sentences, find_announcements, summarize, render

# ---------- Text helpers ----------
const STOPWORDS = Set(split("""
a an the and or but if so to of in on at by for with from as is are was were be been being
it its this that these those i you he she we they me him her us them my your our their
do does did have has had not no yes can could will would should may might must just very
really also about into over then than there here what which who whom when where why how
all any each some such only own same too more most other again further once let lets get
got going go gonna um uh like okay alright right well thing things kind sort today now one
two up down out off see know think say said way make made lot little bit actually basically
"""))

"Split text into sentences."
function split_sentences(text::AbstractString)
    text = replace(text, r"\s+" => " ")
    parts = split(text, r"(?<=[.!?])\s+")
    return [String(strip(p)) for p in parts if length(strip(p)) > 3]
end

"Lowercase words with stopwords removed."
function tokenize(s::AbstractString)
    words = [lowercase(m.match) for m in eachmatch(r"[A-Za-z][A-Za-z'-]+", s)]
    return [w for w in words if !(w in STOPWORDS) && length(w) > 2]
end

# ---------- Announcements and dates ----------
const MONTHS   = "january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec"
const WEEKDAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
const DATE_PATTERNS = [
    Regex("\\b(?:$WEEKDAYS),?\\s+(?:$MONTHS)\\.?\\s+\\d{1,2}(?:st|nd|rd|th)?\\b", "i"),
    Regex("\\b(?:$MONTHS)\\.?\\s+\\d{1,2}(?:st|nd|rd|th)?\\b", "i"),
    r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
    Regex("\\b(?:next|this)\\s+(?:$WEEKDAYS|week|class)\\b", "i"),
    Regex("\\b(?:$WEEKDAYS)\\b", "i"),
    r"\b(?:tomorrow|tonight|by midnight)\b"i,
]

"Classify actual course logistics; topic vocabulary alone is not an announcement."
function announcement_type(s::AbstractString)
    occursin(r"\b(cancel+ed|no class|class is off)\b"i, s) && return "schedule_change"
    (occursin(r"\b(due|deadline|submit|turn in|hand in)\b"i, s) &&
     !occursin(r"\bdue to\b"i, s)) && return "deadline"
    (occursin(r"\b(homework|assignment|project|lab|paper|essay)\b"i, s) &&
     occursin(r"\b(posted|assigned|available)\b"i, s)) && return "deadline"
    exam = occursin(r"\b(exam|midterm|quiz)\b|\bfinal\s+exam\b|\btest\b"i, s)
    exam && (!isempty(find_dates(s)) || occursin(r"\b(upcoming|scheduled|study for|prepare for|covers? chapters?)\b"i, s)) && return "test"
    occursin(r"\b(office hours|next class|next lecture|class next|read (chapter|section|pages?))\b"i, s) && return "upcoming_class"
    return nothing
end

"Pull out every date mention in a sentence (longest patterns first, no overlaps)."
function find_dates(s::AbstractString)
    found = String[]
    taken = falses(ncodeunits(s))
    for pat in DATE_PATTERNS, m in eachmatch(pat, s)
        r = m.offset:(m.offset + ncodeunits(m.match) - 1)
        any(taken[r]) && continue
        taken[r] .= true
        push!(found, m.match)
    end
    return found
end

"Remove spoken lead-ins so a sentence reads like a note."
function clean_note(s::AbstractString)
    s = replace(s, r"^(so|and|also|okay|ok|alright|now|well|um|uh|moving on|next topic is|next topic|let's talk about|quick reminder before i forget|quick reminder)[, ]+"i => "")
    return uppercasefirst(String(strip(s)))
end

"""
Return a vector of (type, detail, date) named tuples, same shape as the LLM output.
"""
function find_announcements(sentences)
    results = NamedTuple{(:type, :detail, :date),Tuple{String,String,Union{String,Nothing}}}[]
    for s in sentences
        occursin(r"\b(see you|that's all|have a good)\b"i, s) && continue   # sign-offs
        kind = announcement_type(s)
        if kind !== nothing
            dates = find_dates(s)
            push!(results, (type = kind, detail = clean_note(s),
                            date = isempty(dates) ? nothing : join(dates, ", ")))
        end
    end
    return results
end

# ---------- Main points and sub-points (TF-IDF + TextTiling) ----------
const TRANSITIONS = r"^(moving on|next topic|next,|now,|now let's|now we|let's move|let's talk|another (important )?(topic|thing)|the next (thing|topic|concept)|switching gears|ok so now|so now)"i

"Bag-of-words for a group of sentences."
function bag(sentences)
    tf = Dict{String,Int}()
    for s in sentences, w in tokenize(s)
        tf[w] = get(tf, w, 0) + 1
    end
    return tf
end

"Cosine similarity between two bags of words."
function cosine(a::Dict, b::Dict)
    (isempty(a) || isempty(b)) && return 0.0
    dot = sum(v * get(b, w, 0) for (w, v) in a)
    na = sqrt(sum(v^2 for v in values(a)))
    nb = sqrt(sum(v^2 for v in values(b)))
    return dot / (na * nb)
end

"""
Group sentences into topic sections (TextTiling).
A boundary is placed where the vocabulary before and after a point stops overlapping,
or where the speaker says a transition phrase like "moving on".
"""
function segment(sentences; window = 3, min_len = 3, max_len = 10)
    n = length(sentences)
    n <= min_len && return [collect(sentences)]

    # similarity across each gap i | i+1
    sim = zeros(n - 1)
    for i in 1:n-1
        left  = bag(sentences[max(1, i - window + 1):i])
        right = bag(sentences[i+1:min(n, i + window)])
        sim[i] = cosine(left, right)
    end

    # depth score: how far each gap dips below the peaks on either side
    depth = zeros(n - 1)
    for i in 1:n-1
        l = sim[i]
        for j in i-1:-1:1
            sim[j] >= l || break
            l = sim[j]
        end
        r = sim[i]
        for j in i+1:n-1
            sim[j] >= r || break
            r = sim[j]
        end
        depth[i] = (l - sim[i]) + (r - sim[i])
    end
    mean_d = sum(depth) / length(depth)
    std_d  = sqrt(sum((depth .- mean_d) .^ 2) / length(depth))
    cutoff = mean_d + 0.5 * std_d

    sections = Vector{Vector{String}}()
    current = String[]
    for (i, s) in enumerate(sentences)
        gap_before = i - 1
        boundary = !isempty(current) && (
            occursin(TRANSITIONS, s) ||
            (gap_before >= 1 && depth[gap_before] > cutoff && length(current) >= min_len) ||
            length(current) >= max_len)
        if boundary
            push!(sections, current); current = String[]
        end
        push!(current, s)
    end
    isempty(current) || push!(sections, current)
    return sections
end

"Inverse document frequency of each word, treating each sentence as a document."
function idf_table(sentences)
    df = Dict{String,Int}()
    for s in sentences, w in Set(tokenize(s))
        df[w] = get(df, w, 0) + 1
    end
    n = length(sentences)
    return Dict(w => log((n + 1) / (c + 1)) + 1 for (w, c) in df)
end

"Score a sentence: sum of TF-IDF of its words, normalized by length."
function score(s, tf, idf)
    toks = tokenize(s)
    isempty(toks) && return 0.0
    return sum(get(tf, w, 0) * get(idf, w, 1.0) for w in toks) / sqrt(length(toks))
end

"Title a section from its top keywords."
function section_title(section, idf)
    tf = bag(section)
    ranked = sort(collect(tf), by = p -> -p[2] * get(idf, p[1], 1.0))
    words = [titlecase(p[1]) for p in ranked[1:min(3, length(ranked))]]
    return join(words, ", ")
end

"""
Extractive summary. Returns a vector of (point, sub_points) named tuples,
same shape as the LLM output. `skip` is a Set of sentences to leave out
(the announcements, which are reported separately).
"""
function summarize(sentences, skip = Set{String}(); subpoints = 3)
    content = [s for s in sentences if !(s in skip) && length(tokenize(s)) >= 3]
    isempty(content) && return NamedTuple{(:point, :sub_points),Tuple{String,Vector{String}}}[]
    idf = idf_table(content)
    out = NamedTuple{(:point, :sub_points),Tuple{String,Vector{String}}}[]
    for section in segment(content)
        tf = bag(section)
        scores = [score(s, tf, idf) for s in section]
        order = sortperm(scores, rev = true)
        main = section[order[1]]
        picks = sort(order[2:min(subpoints + 1, length(order))])   # keep lecture order
        title = section_title(section, idf)
        push!(out, (point = "$title: $(clean_note(main))", sub_points = clean_note.(section[picks])))
    end
    return out
end

# ---------- Full notes from raw text ----------
"Run the whole pure-Julia pipeline and return (title, main_points, announcements)."
function analyze(text::AbstractString)
    sentences = [s for s in split_sentences(text) if !occursin(r"\b(see you|that's all|have a good)\b"i, s)]
    announcements = find_announcements(sentences)
    skip = Set(s for s in sentences if announcement_type(s) !== nothing)
    points = summarize(sentences, skip)
    return (title = "Lecture Notes (offline mode)", main_points = points, announcements = announcements)
end

const ICON = Dict("test" => "[TEST]", "deadline" => "[DUE]", "schedule_change" => "[SCHEDULE]", "upcoming_class" => "[NEXT CLASS]")

"Render notes (from either pipeline) as Markdown."
function render(notes)
    io = IOBuffer()
    println(io, "# ", notes.title, "\n")
    println(io, "## Important Dates & Announcements")
    isempty(notes.announcements) && println(io, "- None mentioned")
    for a in notes.announcements
        d = a.date === nothing ? "" : "  **[$(a.date)]**"
        println(io, "- ", get(ICON, String(a.type), "-"), " ", a.detail, d)
    end
    println(io, "\n## Main Points")
    for (i, p) in enumerate(notes.main_points)
        println(io, "\n### $i. ", p.point)
        for s in p.sub_points
            println(io, "- ", s)
        end
    end
    return String(take!(io))
end

render(text::AbstractString) = render(analyze(text))

end # module
