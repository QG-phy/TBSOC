# Icon Setup for TBSOC

This directory contains the application icons generated from `tbsoc_logo.png`.

## Files

- **icon.ico**: Windows application icon (multi-resolution)
- **icon.icns**: macOS application icon (multi-resolution)

## Regenerating Icons

If you need to update the icons from a new source image:

1. Replace `tbsoc_logo.png` in the root directory
2. Run the icon generator:
   ```bash
   uv run python generate_icons.py
   ```

This will regenerate both `.ico` and `.icns` files with the proper resolutions for each platform.

## Icon Sizes

### Windows (.ico)
- 16x16, 32x32, 48x48, 64x64, 128x128, 256x256

### macOS (.icns)
- 16x16, 32x32, 64x64, 128x128, 256x256, 512x512, 1024x1024 (with @2x variants)

## Usage in PyInstaller

The icons are automatically referenced in `TBSOC.spec`:
- Windows: `icon='resources/icon.ico'` in `EXE()`
- macOS: `icon='resources/icon.icns'` in `BUNDLE()`
