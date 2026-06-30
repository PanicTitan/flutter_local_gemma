---
name: qr-code
description: Generate a QR code for a given text or URL
parameters: {"type": "object", "properties": {"text": {"type": "string", "description": "The text or URL to encode in the QR code"}}}
---

Please use this skill to generate a QR code. Pass the text or url as JSON:
```json
{
  "text": "https://google.com"
}
```
