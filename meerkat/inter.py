import os
import sys
import curses

def custom_menu(my_choices):
	screen = curses.initscr()
	curses.start_color()
	screen.clear()
	curses.init_pair(1, curses.COLOR_RED, curses.COLOR_WHITE)
	screen.keypad(True)
	pos = 1
	x = None
	h = curses.color_pair(1)
	n = curses.A_NORMAL
	choices = ["Zero"]
	choices.extend(my_choices)
	row_base, col_base = 6, 4
	while x != ord('\n'):
		screen.clear()
		screen.border(0)
		screen.addstr(2, 2, "TRANSACTION VIEWER", curses.A_STANDOUT)
		screen.addstr(4, 2, "Please select a label...", curses.A_BOLD)
		screen.addstr(5, 2, "Position is " + str(pos), curses.A_BOLD)

		for i in range(1, len(choices)):
			if pos == i:
				screen.addstr(row_base + i, col_base, str(i) + " -> " + choices[i], h)
			else:
				screen.addstr(row_base + i, col_base, str(i) + " -> " + choices[i], n)

		screen.refresh()
		x = screen.getch()

		direct = False
		for i in range(1, len(choices)):
			if x == ord(str(i)):
				pos = i
				direct = True
				break

		if not direct:
			if x == curses.KEY_DOWN:
				if pos < len(my_choices):
					pos += 1
				else:
					curses.flash()
			elif x == curses.KEY_UP:
				if pos > 1:
					pos -= 1
				else:
					curses.flash()
			elif x != ord('\n'):
				curses.flash()
			else:
				pass
	return pos, ord(str(pos))

if __name__ == '__main__':
	choices = ["One", "Two", "Three", "Four"]
	x, y = custom_menu(choices)
	curses.endwin()
	print("Your selection was {0}".format(x))





