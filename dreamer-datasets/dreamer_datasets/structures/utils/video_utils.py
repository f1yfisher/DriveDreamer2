import imageio
from PIL import Image


def load_video(video_path, num_frames=0):
    reader = imageio.get_reader(video_path)
    if num_frames > 0:
        num_frames = min(reader.count_frames(), num_frames)
    else:
        num_frames = reader.count_frames()
    frames = []
    for i in range(num_frames):
        try:
            frame = Image.fromarray(reader.get_data(i))
            frames.append(frame)
        except Exception:
            continue
    return frames
