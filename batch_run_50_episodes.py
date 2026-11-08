"""
Batch Runner for Kuku Problem 1: Find the Promos
================================================
Processes all 50 episodes in the input pack using the Two-Stage Hybrid Pipeline.
Generates:
  1. raw_outputs/episode_XXX_promos.json (individual episode raw outputs)
  2. raw_outputs/all_episodes_top_promos.json (consolidated master JSON)
  3. raw_outputs/all_episodes_top_promos.csv (consolidated master CSV)
"""

import os
import re
import csv
import json
import glob

# Tiered keyword lexicon
KEYWORDS_S_TIER = {
    "slap", "marry", "wedding", "groom", "bride", "betray", "tears",
    "kill", "shock", "divorce", "bastard", "mistress", "cheating",
    "pregnant", "poison", "blood", "gun", "arrest", "ruin", "destroy"
}
KEYWORDS_A_TIER = {
    "liar", "shame", "humiliation", "kneel", "beg", "plead", "furious",
    "outrage", "disown", "scream", "weep", "rage", "revenge", "heir",
    "billionaire", "imposter", "scandal", "secret", "reveal", "truth",
    "worthless", "refuse"
}
KEYWORDS_B_TIER = {
    "desperate", "cold", "cruel", "heartbroken", "choke", "grab", "gasp",
    "swore", "vow", "promise", "money", "mine", "coward", "dare"
}

def parse_time_str(ts_str):
    clean = ts_str.strip().strip("()")
    parts = clean.split(":")
    try:
        if len(parts) == 2:
            return float(parts[0]) * 60.0 + float(parts[1])
        elif len(parts) == 3:
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
        return float(parts[0])
    except Exception:
        return 0.0

def format_time_str(sec):
    if sec < 0:
        sec = 0.0
    m = int(sec // 60)
    s = sec % 60
    return f"{m:02d}:{s:05.2f}"

def parse_screenplay(raw_text):
    lines = raw_text.splitlines()
    metadata = {
        "episode": "Unknown Episode",
        "logline": "No logline found.",
        "continuity": "No continuity notes found."
    }

    for line in lines[:15]:
        t = line.strip()
        if t.startswith("EPISODE NUMBER:"):
            metadata["episode"] = t.split(":", 1)[1].strip()
        elif t.startswith("EPISODE LOGLINE:"):
            metadata["logline"] = t.split(":", 1)[1].strip()
        elif t.startswith("CONTINUITY NOTES:"):
            metadata["continuity"] = t.split(":", 1)[1].strip()

    time_pattern = re.compile(r"Start:\s*\(?([0-9:.]+)\)?\s*\|\s*End:\s*\(?([0-9:.]+)\)?", re.IGNORECASE)
    block_header_pattern = re.compile(r"^\*\*([^*]+)\*\*:?$")
    scene_pattern = re.compile(r"^SCENE\s+(\d+)", re.IGNORECASE)
    heading_pattern = re.compile(r"^Scene heading:\s*(.+)$", re.IGNORECASE)

    beats = []
    current_scene = "SCENE 01"
    current_heading = "UNTITLED SCENE"
    i = 0
    n = len(lines)
    time_cursor = 0.0

    while i < n:
        line = lines[i].strip()
        m_scene = scene_pattern.match(line)
        if m_scene:
            current_scene = f"SCENE {m_scene.group(1).zfill(2)}"
            i += 1
            continue

        m_heading = heading_pattern.match(line)
        if m_heading:
            current_heading = m_heading.group(1).strip()
            i += 1
            continue

        m_header = block_header_pattern.match(line)
        if m_header:
            header_raw = m_header.group(1).strip()
            i += 1

            start_sec = None
            end_sec = None
            start_str = ""
            end_str = ""
            content_lines = []

            while i < n:
                sub = lines[i].strip()
                if block_header_pattern.match(sub) or scene_pattern.match(sub):
                    break
                if sub:
                    m_time = time_pattern.search(sub)
                    if m_time:
                        start_str = m_time.group(1)
                        end_str = m_time.group(2)
                        start_sec = parse_time_str(start_str)
                        end_sec = parse_time_str(end_str)
                    else:
                        content_lines.append(sub)
                i += 1

            content_text = " ".join(content_lines).strip()
            if start_sec is None or end_sec is None:
                w_count = len(content_text.split())
                est_dur = max(2.0, w_count / 2.5)
                start_sec = time_cursor
                end_sec = start_sec + est_dur
                start_str = format_time_str(start_sec)
                end_str = format_time_str(end_sec)

            time_cursor = max(time_cursor, end_sec)

            header_upper = header_raw.upper()
            is_camera = "CAMERA" in header_upper
            is_action = "ACTION" in header_upper
            is_sound = "SOUND FX" in header_upper or "SFX" in header_upper
            is_dialogue = not (is_camera or is_action or is_sound)

            speaker = header_raw
            if is_dialogue and "(" in header_raw and ")" in header_raw:
                speaker = header_raw[:header_raw.find("(")].strip()

            beats.append({
                "scene": current_scene,
                "heading": current_heading,
                "header": header_raw,
                "speaker": speaker,
                "is_dialogue": is_dialogue,
                "is_action": is_action,
                "is_camera": is_camera,
                "is_sound": is_sound,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start_str": start_str,
                "end_str": end_str,
                "text": content_text
            })
            continue

        i += 1

    return metadata, beats

def score_window(beats, start_sec, end_sec):
    full_text = " ".join(b["text"] for b in beats).lower()
    full_headers = " ".join(b["header"] for b in beats).lower()
    combined = full_text + " " + full_headers

    matched_kws = []
    kw_score = 0.0

    for kw in KEYWORDS_S_TIER:
        c = len(re.findall(r"\b" + kw + r"\b", combined))
        if c > 0:
            kw_score += c * 3.0
            matched_kws.append(f"{kw} (x{c})")

    for kw in KEYWORDS_A_TIER:
        c = len(re.findall(r"\b" + kw + r"\b", combined))
        if c > 0:
            kw_score += c * 2.0
            matched_kws.append(f"{kw} (x{c})")

    for kw in KEYWORDS_B_TIER:
        c = len(re.findall(r"\b" + kw + r"\b", combined))
        if c > 0:
            kw_score += c * 1.0
            matched_kws.append(f"{kw} (x{c})")

    # Hook bonus (0-4s)
    hook_beats = [b for b in beats if b["start_sec"] <= start_sec + 4.0]
    hook_text = " ".join(b["text"] for b in hook_beats)
    hook_score = 0.0
    if any(b["is_dialogue"] for b in hook_beats):
        hook_score += 4.0
    if any(b["is_action"] and any(k in b["text"].lower() for k in ["slap", "grab", "push", "kneel"]) for b in hook_beats):
        hook_score += 5.0
    if any(p in hook_text for p in ["?", "!", "Why", "Stop", "Turn", "Who", "Look"]):
        hook_score += 2.0

    # Cliffhanger bonus (end-4s)
    cliff_beats = [b for b in beats if b["end_sec"] >= end_sec - 4.0]
    cliff_text = " ".join(b["text"] for b in cliff_beats).strip()
    cliff_score = 0.0
    if any(b["is_dialogue"] for b in cliff_beats):
        cliff_score += 3.0
    if cliff_text.endswith("?") or cliff_text.endswith("!") or cliff_text.endswith("..."):
        cliff_score += 4.0
    if any(k in cliff_text.lower() for k in ["marry", "betray", "never", "cannot", "mine", "truth", "who", "father", "died"]):
        cliff_score += 3.0

    # Dialogue ratio
    dialogue_beats = sum(1 for b in beats if b["is_dialogue"])
    total_beats = len(beats)
    dialogue_score = (dialogue_beats / total_beats * 4.0) if total_beats > 0 else 0.0

    total = round(kw_score + hook_score + cliff_score + dialogue_score, 2)
    return total, matched_kws

def extract_candidates(beats, min_dur=20.0, max_dur=60.0, top_k=10, iou_thresh=0.65):
    raw = []
    n = len(beats)
    for i in range(n):
        s_beat = beats[i]
        s_sec = s_beat["start_sec"]
        for j in range(i, n):
            e_beat = beats[j]
            e_sec = e_beat["end_sec"]
            dur = e_sec - s_sec
            if dur > max_dur:
                break
            if dur >= min_dur:
                w_beats = beats[i:j+1]
                score, kws = score_window(w_beats, s_sec, e_sec)
                raw.append({
                    "start_sec": s_sec,
                    "end_sec": e_sec,
                    "start_str": format_time_str(s_sec),
                    "end_str": format_time_str(e_sec),
                    "duration": round(dur, 2),
                    "score": score,
                    "keywords": kws,
                    "beats": w_beats
                })

    raw.sort(key=lambda x: x["score"], reverse=True)
    selected = []
    for cand in raw:
        overlap = False
        for s in selected:
            inter = max(0.0, min(cand["end_sec"], s["end_sec"]) - max(cand["start_sec"], s["start_sec"]))
            min_l = min(cand["duration"], s["duration"])
            if min_l > 0 and (inter / min_l) > iou_thresh:
                overlap = True
                break
        if not overlap:
            selected.append(cand)
            if len(selected) >= top_k:
                break
    return selected

def evaluate_and_rank_candidates(metadata, candidates, episode_num_str):
    results = []
    for idx, cand in enumerate(candidates, 1):
        promo_id = f"{episode_num_str}_PROMO_{idx:02d}"
        beats = cand["beats"]
        start_sec = cand["start_sec"]
        end_sec = cand["end_sec"]

        # Hook analysis
        first_beats = [b for b in beats if b["start_sec"] <= start_sec + 4.0]
        first_diag = next((b for b in first_beats if b["is_dialogue"]), None)
        first_act = next((b for b in first_beats if b["is_action"]), None)

        if first_diag:
            hook_text = f'{first_diag["speaker"]}: "{first_diag["text"][:75]}"'
            hook_rating = 5 if any(k in first_diag["text"].lower() for k in ["turn", "stop", "why", "who", "marry", "liar", "bastard"]) else 4
        elif first_act:
            hook_text = f'Action: {first_act["text"][:85]}'
            hook_rating = 4
        else:
            hook_text = f'Scene Opening: {beats[0]["text"][:80]}'
            hook_rating = 3

        # Cliffhanger analysis
        last_beats = [b for b in beats if b["end_sec"] >= end_sec - 4.0]
        last_diag = next((b for b in reversed(last_beats) if b["is_dialogue"]), None)

        if last_diag:
            cliff_text = f'{last_diag["speaker"]}: "{last_diag["text"][:75]}"'
            has_p = any(last_diag["text"].strip().endswith(p) for p in ["?", "!", "..."])
            cliff_rating = 5 if has_p else 4
        else:
            cliff_text = f'Tense cut: {beats[-1]["text"][:80]}'
            cliff_rating = 3

        # Standalone comprehension
        kw_len = len(cand["keywords"])
        if kw_len >= 4:
            comp = "High"
            comp_num = 5.0
        elif kw_len >= 2:
            comp = "Medium"
            comp_num = 3.0
        else:
            comp = "Low"
            comp_num = 1.0

        top_kws = ", ".join(cand["keywords"][:3]) if cand["keywords"] else "intense confrontation"
        why_it_works = f"High emotional confrontation driven by {top_kws} with a tense unresolved cliffhanger."

        # Composite score
        base = (hook_rating * 0.4) + (cliff_rating * 0.4) + (comp_num * 0.2)
        viral_score = round(base * 2.0, 2)

        results.append({
            "episode": metadata["episode"],
            "promo_id": promo_id,
            "start_time": cand["start_str"],
            "end_time": cand["end_str"],
            "duration_sec": cand["duration"],
            "hook_text": hook_text,
            "hook_rating": hook_rating,
            "cliffhanger_text": cliff_text,
            "cliffhanger_rating": cliff_rating,
            "standalone_comprehension": comp,
            "why_it_works": why_it_works,
            "viral_score": viral_score
        })

    results.sort(key=lambda x: (x["viral_score"], x["hook_rating"]), reverse=True)
    return results

def main():
    os.makedirs("raw_outputs", exist_ok=True)
    files = sorted(glob.glob("episode-*.txt"))
    print(f"Found {len(files)} episode files to process.")

    all_promos = []

    for f_idx, file_path in enumerate(files, 1):
        filename = os.path.basename(file_path)
        ep_match = re.search(r"episode-(\d+)", filename, re.IGNORECASE)
        ep_tag = f"EP{int(ep_match.group(1)):03d}" if ep_match else f"EP{f_idx:03d}"

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        metadata, beats = parse_screenplay(content)
        candidates = extract_candidates(beats, min_dur=20.0, max_dur=60.0, top_k=5, iou_thresh=0.65)
        evals = evaluate_and_rank_candidates(metadata, candidates, ep_tag)

        # Write episode JSON
        ep_out_path = os.path.join("raw_outputs", f"{ep_tag.lower()}_promos.json")
        with open(ep_out_path, "w", encoding="utf-8") as out_f:
            json.dump({
                "episode": metadata["episode"],
                "logline": metadata["logline"],
                "total_promos": len(evals),
                "promos": evals
            }, out_f, indent=2)

        all_promos.extend(evals)
        if f_idx % 10 == 0 or f_idx == len(files):
            print(f"[{f_idx}/{len(files)}] Processed {metadata['episode']} -> {len(evals)} promos extracted.")

    # Master files
    all_promos.sort(key=lambda x: x["viral_score"], reverse=True)

    master_json_path = os.path.join("raw_outputs", "all_episodes_top_promos.json")
    with open(master_json_path, "w", encoding="utf-8") as out_f:
        json.dump(all_promos, out_f, indent=2)

    master_csv_path = os.path.join("raw_outputs", "all_episodes_top_promos.csv")
    fieldnames = [
        "episode", "promo_id", "start_time", "end_time", "duration_sec",
        "hook_rating", "hook_text", "cliffhanger_rating", "cliffhanger_text",
        "standalone_comprehension", "why_it_works", "viral_score"
    ]
    with open(master_csv_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        for p in all_promos:
            writer.writerow(p)

    print(f"\nSUCCESS! Processed {len(files)} episodes.")
    print(f"Generated {len(all_promos)} total ranked promos across all episodes.")
    print(f"Master CSV saved to: {master_csv_path}")
    print(f"Master JSON saved to: {master_json_path}")
    print(f"Top #1 Promo overall: {all_promos[0]['promo_id']} from {all_promos[0]['episode']} with Viral Score {all_promos[0]['viral_score']}/10")

if __name__ == "__main__":
    main()
