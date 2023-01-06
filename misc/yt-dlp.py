# usage: to let this program keep running after logging out over ssh, run
# `nohup python yt-dlp.py &`. stdout will redirect to `nohup.out`

import subprocess


# list of urls of youtube playlists to download
playlistUrls = [
    "https://www.youtube.com/playlist?list=OLAK5uy_m6DPmzLVYAE6a9OKViK3HM7nxcp6v4ORE",
    "https://www.youtube.com/playlist?list=PL8F6B0753B2CCA128",
    "https://www.youtube.com/playlist?list=PLbYGZP48h5lLUSCqvU6vuarjEaQWM_dcN",
    "https://www.youtube.com/playlist?list=PLZN_exA7d4RVmCQrG5VlWIjMOkMFZVVOc",
    "https://www.youtube.com/playlist?list=PL3oW2tjiIxvQW6c-4Iry8Bpp3QId40S5S",
    "https://www.youtube.com/playlist?list=PLetgZKHHaF-Zq1Abh-ZGC4liPd_CV3Uo4",
    "https://www.youtube.com/playlist?list=PLFYZh4cL2f0qohRXux42oPNysXzqkT3m_",
    "https://www.youtube.com/playlist?list=PLw-VjHDlEOgs658kAHR_LAaILBXb-s6Q5",
]


# formula for how to save downloaded videos to disk
outputNameFormula = r"%(playlist)s/%(playlist_index)s/%(title)s.%(ext)s"

# directory to save videos to
# memstar:
# outputDirectory = "/z/atrom/datasets/unlabeled/YouTube"
# qubit1:
outputDirectory = "/home/m20adams/atrom/datasets/unlabeled/YouTube"

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
    
print("Finished")

