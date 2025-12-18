# macOS Installation Guide

## ⚠️ macOS Security Warning

When you first open TBSOC on macOS, you may see this warning:

> "TBSOC" cannot be opened because Apple cannot verify that it is free from malware.

**This is normal!** The app is not signed with an Apple Developer certificate because it's open-source software.

## ✅ How to Open TBSOC Safely

### Method 1: Right-Click to Open (Recommended)
1. Locate `TBSOC.app` in your Applications folder
2. **Right-click** (or Control-click) on the app
3. Select **"Open"** from the menu
4. Click **"Open"** in the dialog that appears
5. The app will now open and remember this choice

### Method 2: System Settings
1. Try to open TBSOC normally (it will be blocked)
2. Go to **System Settings** → **Privacy & Security**
3. Scroll down to the **Security** section
4. Click **"Open Anyway"** next to the TBSOC message
5. Click **"Open"** to confirm

### Method 3: Command Line (Advanced)
Open Terminal and run:
```bash
xattr -cr /Applications/TBSOC.app
```

Then open the app normally.

## 🔒 Why This Happens

macOS Gatekeeper blocks apps that aren't:
- Signed with an Apple Developer certificate ($99/year)
- Notarized by Apple (requires signing)

Since TBSOC is free and open-source, we don't have Apple Developer signing. The app is safe to use - you can verify the source code on GitHub!

## 🛡️ For Developers: Code Signing (Optional)

If you want to distribute signed builds:

1. **Get an Apple Developer Account** ($99/year)
2. **Update the workflow** to include signing:

```yaml
- name: Sign macOS app
  env:
    MACOS_CERTIFICATE: ${{ secrets.MACOS_CERTIFICATE }}
    MACOS_CERTIFICATE_PWD: ${{ secrets.MACOS_CERTIFICATE_PWD }}
  run: |
    # Import certificate
    echo $MACOS_CERTIFICATE | base64 --decode > certificate.p12
    security create-keychain -p actions temp.keychain
    security import certificate.p12 -k temp.keychain -P $MACOS_CERTIFICATE_PWD -T /usr/bin/codesign
    security set-key-partition-list -S apple-tool:,apple: -s -k actions temp.keychain
    
    # Sign the app
    codesign --deep --force --verify --verbose --sign "Developer ID Application: YOUR NAME" dist/TBSOC.app

- name: Notarize app
  run: |
    # Submit for notarization
    xcrun notarytool submit TBSOC-macOS.dmg --apple-id YOUR_EMAIL --password APP_SPECIFIC_PASSWORD --team-id TEAM_ID --wait
    
    # Staple the notarization ticket
    xcrun stapler staple TBSOC-macOS.dmg
```

## 📚 More Information

- [Apple Gatekeeper Documentation](https://support.apple.com/en-us/HT202491)
- [Notarizing macOS Software](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution)
