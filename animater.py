import numpy as np
from PIL import Image
from collections import OrderedDict
from multiprocessing import Pool
from tqdm import tqdm
import math

LEGACY_PALETTE = {
    'darkcyan': '#0074BA',
    'dodgerblue': '#007ACF',
    'teal': '#008463',
    'deepskyblue': '#00A6ED',
    'springgreen': '#00D26A',
    'mediumspringgreen': '#00F397',
    'darkslateblue': '#1345B7',
    'black': '#1C1C1C',
    'royalblue': '#246ADE',
    'aqua': '#26EAFC',
    'darkslategray': '#28262F',
    'midnightblue': '#321B41',
    'mediumturquoise': '#37B9F1',
    'forestgreen': '#44911B',
    'cornflowerblue': '#46A4FB',
    'turquoise': '#50E2FF',
    'mediumseagreen': '#5AB557',
    'lightskyblue': '#5DD7FD',
    'dimgray': '#636363',
    'aquamarine': '#6AFCAC',
    'saddlebrown': '#6D4534',
    'slategrey': '#6E649B',
    'sienna': '#784B39',
    'slateblue': '#785DC8',
    'yellowgreen': '#79BB48',
    'blueviolet': '#803BFF',
    'mediumpurple': '#866ECE',
    'darkorchid': '#8C42B2',
    'darkgrey': '#998EA4',
    'lightblue': '#9DDAF2',
    'peru': '#A38439',
    'indianred': '#A56953',
    'darkseagreen': '#A6CB93',
    'lightsteelblue': '#AC9DD3',
    'paleturquoise': '#AEDDFF',
    'silver': '#B5B1CC',
    'mediumorchid': '#B859D3',
    'chocolate': '#B97028',
    'mediumvioletred': '#BB1D80',
    'darkkhaki': '#BFCC82',
    'greenyellow': '#C3EF3C',
    'rosybrown': '#C68D7B',
    'plum': '#C790F1',
    'lightgrey': '#C7C5D3',
    'crimson': '#CA0B4A',
    'thistle': '#CDC4D6',
    'lightcyan': '#D2F4FE',
    'lightgray': '#D3D3D3',
    'palegoldenrod': '#D5F1B7',
    'gainsboro': '#D8D8D8',
    'khaki': '#DCF68F',
    'darksalmon': '#E39D89',
    'burlywood': '#E3B279',
    'goldenrod': '#E4A33C',
    'lavender': '#E7E7E7',
    'lightpink': '#E895B0',
    'aliceblue': '#E8F9FF',
    'honeydew': '#EAFBF3',
    'whitesmoke': '#EEEBF0',
    'azure': '#EEFEF8',
    'sandybrown': '#F0B362',
    'pink': '#F1BECF',
    'linen': '#F1EBE9',
    'salmon': '#F37366',
    'tomato': '#F46A33',
    'orange': '#F59F00',
    'deeppink': '#F70A8D',
    'antiquewhite': '#F7E5D0',
    'ghostwhite': '#F7F6FC',
    'lavenderblush': '#F9EEF2',
    'lightcoral': '#FA557D',
    'bisque': '#FBE3CA',
    'mistyrose': '#FCDEE4',
    'moccasin': '#FDE8B5',
    'oldlace': '#FDF3E4',
    'seashell': '#FDF7EF',
    'hotpink': '#FE63AE',
    'navajowhite': '#FED9A5',
    'floralwhite': '#FEFAF2',
    'darkorange': '#FF8101',
    'coral': '#FF822D',
    'peachpuff': '#FFD7C2',
    'papayawhip': '#FFEFD4',
    'white': '#FFFFFF',
}
EMOJI_MAP = {
    'darkcyan': '💙',
    'dodgerblue': '🦸',
    'teal': '🐢',
    'deepskyblue': '🫃',
    'springgreen': '🤢',
    'mediumspringgreen': '🦠',
    'darkslateblue': '⛈',
    'black': '⬛',
    'royalblue': '🏙',
    'aqua': '🚉',
    'darkslategray': '🌃',
    'midnightblue': '🕳',
    'mediumturquoise': '❄',
    'forestgreen': '🧝',
    'cornflowerblue': '🚏',
    'turquoise': '🛤',
    'mediumseagreen': '🫛',
    'lightskyblue': '🛴',
    'dimgray': '🦍',
    'aquamarine': '🩳',
    'saddlebrown': '🤎',
    'slategrey': '🎸',
    'sienna': '🌻',
    'slateblue': '🔙',
    'yellowgreen': '🪇',
    'blueviolet': '🫸',
    'mediumpurple': '➰',
    'darkorchid': '🍇',
    'darkgrey': '🏥',
    'lightblue': '🚴',
    'peru': '🌠',
    'indianred': '🐴',
    'darkseagreen': '🌾',
    'lightsteelblue': '💷',
    'paleturquoise': '🐳',
    'silver': '💱',
    'mediumorchid': '🧤',
    'chocolate': '🧌',
    'mediumvioletred': '🤩',
    'darkkhaki': '🍈',
    'greenyellow': '🍐',
    'rosybrown': '🈚',
    'plum': '🟪',
    'lightgrey': '🚴',
    'crimson': '👹',
    'thistle': '🤖',
    'lightcyan': '🪪',
    'lightgray': '🎅',
    'palegoldenrod': '🦗',
    'gainsboro': '◻',
    'khaki': '🖍',
    'darksalmon': '🙉',
    'burlywood': '💴',
    'goldenrod': '🪙',
    'lavender': '🚬',
    'lightpink': '🏎',
    'aliceblue': '🚡',
    'honeydew': '🚵',
    'whitesmoke': '🐧',
    'azure': '🏡',
    'sandybrown': '🪗',
    'pink': '👙',
    'linen': '⛪',
    'salmon': '🏜',
    'tomato': '🏮',
    'orange': '🧘',
    'deeppink': '🤪',
    'antiquewhite': '📋',
    'ghostwhite': '®',
    'lavenderblush': '🪭',
    'lightcoral': '🈹',
    'bisque': '🛞',
    'mistyrose': '🤡',
    'moccasin': '⚖',
    'oldlace': '🧎',
    'seashell': '🧇',
    'hotpink': '🪅',
    'navajowhite': '🥓',
    'floralwhite': '🧚',
    'darkorange': '🙎',
    'coral': '🤭',
    'peachpuff': '📴',
    'papayawhip': '🦒',
    'white': '⬜',
}

def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    h = hex_str.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def color_distance_sq(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> int:
    return sum((a - b) ** 2 for a, b in zip(c1, c2))

def nearest_legacy_color(hex_value: str) -> str:
    target = hex_to_rgb(hex_value)
    best_name, best_dist = None, math.inf
    for name, hx in LEGACY_PALETTE.items():
        d = color_distance_sq(target, hex_to_rgb(hx))
        if d < best_dist:
            best_dist, best_name = d, name
    return best_name

def hex2emoji(hex_value: str) -> str:
    name = nearest_legacy_color(hex_value)
    return EMOJI_MAP.get(name, '⬜')

def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return '#{:02X}{:02X}{:02X}'.format(*rgb)

def process_block(args):
    block, x, y, debug_positions = args
    avg_rgb = tuple(np.mean(block.reshape(-1, 3), axis=0).astype(int))
    hex_color = rgb_to_hex(avg_rgb)
    name = nearest_legacy_color(hex_color)
    emoji = EMOJI_MAP.get(name, '⬜')
    # Debug print for selected blocks
    if (x, y) in debug_positions:
        print(f"[DEBUG] Block ({x},{y}) avg={avg_rgb} hex={hex_color} → {name} → {emoji}")
    return (y, x, emoji)

def image_to_emoji_art(image_path: str,
                       block_size: int = 12,
                       quantize: bool = True,
                       debug_samples: int = 3) -> str:
    img = Image.open(image_path).convert('RGB')

    # Optionally quantize
    if quantize:
        img = img.quantize(colors=128).convert('RGB')

    arr = np.array(img)
    h, w, _ = arr.shape

    # Pick a few random block origins to debug
    ys = np.random.choice(range(0, h, block_size), size=debug_samples, replace=False)
    xs = np.random.choice(range(0, w, block_size), size=debug_samples, replace=False)
    debug_positions = set(zip(xs, ys))

    tasks = [
        (arr[y:y+block_size, x:x+block_size], x, y, debug_positions)
        for y in range(0, h, block_size)
        for x in range(0, w, block_size)
    ]

    results = []
    with Pool() as pool:
        for res in tqdm(pool.imap_unordered(process_block, tasks),
                        total=len(tasks),
                        desc="Processing blocks",
                        unit=' Blocks'
                        ):
            results.append(res)

    # Reassemble lines
    lines_by_y = {}
    for y, x, em in results:
        lines_by_y.setdefault(y, []).append((x, em))

    output = []
    for y in sorted(lines_by_y):
        row = ''.join(em for x, em in sorted(lines_by_y[y]))
        output.append(row)

    return "\n".join(output)

if __name__ == '__main__':
    # Prompt for image file path
    image_path = input("Enter the path to your image file: ").strip()
    while not image_path:
        print("  → Please enter a non-empty path.")
        image_path = input("Enter the path to your image file: ").strip()

    # Prompt for block size
    while True:
        size_str = input("Enter block size (integer, smaller = higher detail; default 12): ").strip()
        if not size_str:
            block_size = 12
            break
        try:
            block_size = int(size_str)
            if block_size <= 0:
                raise ValueError
            break
        except ValueError:
            print("  → Invalid input; please enter a positive integer.")

    # Prompt whether to quantize
    quantize_in = input("Quantize palette? (Y/n): ").strip().lower()
    quantize = not (quantize_in == 'n')

    # Run the conversion
    art = image_to_emoji_art(
        image_path,
        block_size=block_size,
        quantize=quantize,
        debug_samples=3
    )

    # Save to HTML
    out_path = "emoji_art.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Emoji Art</title>
<style>
  body {{ background: #000; color: #fff; }}
  pre {{ font-family: monospace; font-size: 10px; line-height: 1; }}
</style></head><body><pre>{art}</pre></body></html>""")

    print(f"\n✅ Done! Your emoji art is saved as '{out_path}'.")