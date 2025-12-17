#!/usr/bin/env python3
"""
Icon Generator for TBSOC
Converts tbsoc_logo.png to .icns (macOS) and .ico (Windows) formats
"""

import os
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)

def generate_icons():
    """Generate .icns and .ico files from tbsoc_logo.png"""
    
    # Paths
    base_dir = Path(__file__).parent
    source_png = base_dir / "tbsoc_logo.png"
    output_dir = base_dir / "resources"
    output_dir.mkdir(exist_ok=True)
    
    if not source_png.exists():
        print(f"Error: {source_png} not found!")
        sys.exit(1)
    
    # Load source image
    img = Image.open(source_png)
    
    # Convert to RGBA if needed
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    print(f"Source image: {source_png}")
    print(f"Original size: {img.size}")
    
    # Generate Windows .ico (multiple sizes)
    ico_path = output_dir / "icon.ico"
    ico_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    
    ico_images = []
    for size in ico_sizes:
        resized = img.resize(size, Image.Resampling.LANCZOS)
        ico_images.append(resized)
    
    ico_images[0].save(
        ico_path,
        format='ICO',
        sizes=ico_sizes,
        append_images=ico_images[1:]
    )
    print(f"✓ Created Windows icon: {ico_path}")
    
    # Generate macOS .icns
    # For .icns, we need to create an iconset directory with specific sizes
    icns_path = output_dir / "icon.icns"
    iconset_dir = output_dir / "icon.iconset"
    iconset_dir.mkdir(exist_ok=True)
    
    # macOS icon sizes (https://developer.apple.com/design/human-interface-guidelines/macos/icons-and-images/app-icon/)
    icns_sizes = [
        (16, 'icon_16x16.png'),
        (32, 'icon_16x16@2x.png'),
        (32, 'icon_32x32.png'),
        (64, 'icon_32x32@2x.png'),
        (128, 'icon_128x128.png'),
        (256, 'icon_128x128@2x.png'),
        (256, 'icon_256x256.png'),
        (512, 'icon_256x256@2x.png'),
        (512, 'icon_512x512.png'),
        (1024, 'icon_512x512@2x.png'),
    ]
    
    for size, filename in icns_sizes:
        resized = img.resize((size, size), Image.Resampling.LANCZOS)
        resized.save(iconset_dir / filename, 'PNG')
    
    print(f"✓ Created iconset: {iconset_dir}")
    
    # Convert iconset to icns using macOS iconutil (if on macOS)
    if sys.platform == 'darwin':
        import subprocess
        try:
            subprocess.run(
                ['iconutil', '-c', 'icns', str(iconset_dir), '-o', str(icns_path)],
                check=True
            )
            print(f"✓ Created macOS icon: {icns_path}")
            
            # Clean up iconset directory
            import shutil
            shutil.rmtree(iconset_dir)
            print(f"✓ Cleaned up temporary iconset directory")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not create .icns file: {e}")
            print(f"Iconset directory preserved at: {iconset_dir}")
    else:
        print(f"⚠ Not on macOS - .icns creation requires iconutil")
        print(f"  Iconset directory created at: {iconset_dir}")
        print(f"  Run 'iconutil -c icns {iconset_dir}' on macOS to create .icns")
    
    print("\n✅ Icon generation complete!")
    print(f"\nGenerated files:")
    print(f"  - {ico_path} (Windows)")
    if icns_path.exists():
        print(f"  - {icns_path} (macOS)")
    else:
        print(f"  - {iconset_dir}/ (macOS iconset - needs iconutil)")

if __name__ == '__main__':
    generate_icons()
