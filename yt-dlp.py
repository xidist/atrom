import subprocess


# list of urls of youtube playlists to download
playlistUrls = [
                "https://www.youtube.com/playlist?list=OLAK5uy_m6DPmzLVYAE6a9OKViK3HM7nxcp6v4ORE",
                "https://www.youtube.com/playlist?list=PL8F6B0753B2CCA128"
                ]


# formula for how to save downloaded videos to disk
outputNameFormula = r"%(playlist)s/%(playlist_index)s/%(title)s.%(ext)s"

# directory to save videos to. should end in a forward slash
outputDirectory = "/z/atrom/datasets/unlabeled/YouTube/"

for url in playlistUrls:
    args = []
    args += ["yt-dlp"]
    # extract audio-only file, and then convert to wav using ffmpeg
    args += ["-x", "--audio-format", "wav"]
    # save videos using our output formula
    args += ["-o", outputDirectory + outputNameFormula]
    args += [url]

    subprocess.run(args)

