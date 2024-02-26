import pyautogui
import time

# Give yourself a few seconds to open a text editor or some other application to see the script in action
time.sleep(5)

# Move the mouse cursor to a specific location on the screen
pyautogui.moveTo(500, 500, duration=1)

# Click at the current mouse location
pyautogui.click()

# Type a string at the current cursor location
pyautogui.write('Hello, world!', interval=0.25)

# You can also press individual keys
pyautogui.press('enter')

# Move mouse to another location
pyautogui.moveTo(600, 500, duration=1)

# Right click at the new location
pyautogui.rightClick()
