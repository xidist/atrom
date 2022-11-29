import subprocess


# list of urls of youtube playlists to download
playlistUrls = [
                "https://www.youtube.com/playlist?list=OLAK5uy_m6DPmzLVYAE6a9OKViK3HM7nxcp6v4ORE",
                "https://www.youtube.com/playlist?list=PL8F6B0753B2CCA128"
                ]


# formula for how to save downloaded videos to disk
outputNameFormula = r"%(playlist)s/%(playlist_index)s/%(title)s.%(ext)s"

# directory to save videos to
outputDirectory = "/z/atrom/datasets/unlabeled/YouTube"

# path to the archive file
downloadArchiveFile = outputDirectory + "/download_archive.txt"

for url in playlistUrls:
    """
    --download-archive: keep a small file around that lets us
      skip redownloading videos in a playlist
      (useful for if we rerun after adding to playlistUrls)
    -f: choose the version of the video with the best audio
      (bouns: often speeds up downloads)
    -x: extract audio-only file
    --audio-format wav: convert to wav using ffmpeg
    -P: save all videos in this directory
    -o: save videos using this formula
    """
    
    args = []
    args += ["yt-dlp"]
    args += ["--download-archive", downloadArchiveFile]    
    args += ["-f", "ba"]
    args += ["-x", "--audio-format", "wav"]
    args += ["-P", outputDirectory]
    args += ["-o", outputNameFormula]
    args += [url]

    subprocess.run(args)

