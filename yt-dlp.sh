# -f ba: choose best audio format
# -x --audio-format wav: extract audio-only file, then convert to wav
# -o "/z/atrom/datasets/unlabeled/YouTube/%(playlist)s/%(playlist_index)s/%(title)s.%(ext)s" : output format for playlists

yt-dlp \
    -f ba \
    -x --audio-format wav \
    -o "/z/atrom/datasets/unlabeled/YouTube/%(playlist)s/%(playlist_index)s/%(title)s.%(ext)s" \
    "https://www.youtube.com/playlist?list=OLAK5uy_m6DPmzLVYAE6a9OKViK3HM7nxcp6v4ORE"
