# Manual UI Test Checklist

## Desktop
- [ ] `python run.py` launches the Flask app without errors.
- [ ] Home page loads with hero text and message input centered.
- [ ] Enter a sample message and submit to reach results page.
- [ ] Predicted categories display as badges with confidence percentages.
- [ ] Confidence bar chart renders and is interactive (hover shows values).
- [ ] "Classify Another Message" button returns to home page.
- [ ] Window resized to 320px width retains readable layout.

## Mobile
- [ ] Load home page in mobile viewport (e.g., browser dev tools) and verify responsive navigation.
- [ ] Input field and submit button fit within viewport without horizontal scrolling.
- [ ] Results page badges wrap cleanly and chart resizes to screen width.
- [ ] Keyboard dismissal returns to full results view.
- [ ] Navigate back to home using button or browser back works correctly.
