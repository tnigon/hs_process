# -*- coding: utf-8 -*-
#from tkinter import Tk, Label, Button, RIGHT
#
from tkinter import Tk, Canvas, Label, Button, Entry, IntVar, END, W, E, N, S, ttk
import numpy as np
from PIL import Image, ImageTk
import random


class AutoScrollbar(ttk.Scrollbar):
    ''' A scrollbar that hides itself if it's not needed.
        Works only if you use the grid geometry manager '''
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
        ttk.Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise Tk.TclError('Cannot use pack with this widget')

    def place(self, **kw):
        raise Tk.TclError('Cannot use place with this widget')


class GUI_image_crop:
    def __init__(self, master):
        self.master = master
        master.title('Crop Image - HS_process')
        self.ul_x = 0
        self.ul_y = 0
        self.entered_number_x = 0
        self.entered_number_y = 0
        self.canvas_id_x1 = None
        self.canvas_id_y1 = None

        text = ('Please choose the upper left pixel of the cropped area (Scroll to zoom)')
        self.label = Label(master, text=text)
        self.label.pack()

        self.text_ul_x = IntVar()
        self.text_ul_y = IntVar()
        self.text_ul_x.set(self.ul_x)
        self.text_ul_y.set(self.ul_y)
        self.label_ul_static_x = Label(master, text='Upper left X pixel: ')
        self.label_ul_static_y = Label(master, text='Upper left Y pixel: ')
        self.label_ul_x = Label(master, textvariable=self.text_ul_x)
        self.label_ul_y = Label(master, textvariable=self.text_ul_y)

        vcmd = master.register(self.validate) # we have to wrap the command
        self.entry_ul_x = Entry(master, validate="key", validatecommand=(vcmd, '%P', 'x'))
        self.entry_ul_y = Entry(master, validate="key", validatecommand=(vcmd, '%P', 'y'))

        self.button_ok = Button(master, text="OK", command=lambda: self.update("OK"))
        self.button_reset = Button(master, text="Reset", command=lambda: self.update("reset"))
#        self.button_ok_ul_y = Button(master, text="OK", command=lambda: self.update("OK"))
#        self.button_reset_ul_y = Button(master, text="Reset", command=lambda: self.update("reset"))

#        array = np.ones((2000, 640))*150
#        a = np.zeros((100, 640,3))
#        a[:,:,0] = 255
#        b = np.zeros((100, 640,3))
#        b[:,:,1] = 255
#        c = np.zeros((100, 1280,3))
#        c[:,:,2] = 255
#        array = np.vstack((c, np.hstack((a, b))))
        array = np.random.randint(255, size=(100,400,3),dtype=np.uint8)
        self.image = Image.fromarray(array, mode='RGB')

#        new_size = int(self.imscale * width), int(self.imscale * height)
#        imagetk = ImageTk.PhotoImage(self.image.resize(new_size))
        vbar = AutoScrollbar(master, orient='vertical')
        hbar = AutoScrollbar(master, orient='horizontal')

#        self.image = ImageTk.PhotoImage(image=Image.fromarray(array))
        self.canvas = Canvas(master, highlightthickness=0,
                             xscrollcommand=hbar.set, yscrollcommand=vbar.set)
#        self.canvas.create_image(10,10, anchor='nw', image=self.image)

        vbar.configure(command=self.canvas.yview)  # bind scrollbars to the canvas
        hbar.configure(command=self.canvas.xview)

        # Bind events to the Canvas
        self.canvas.bind('<ButtonPress-1>', self.click)
        self.canvas.bind('<ButtonPress-2>', self.move_from)
        self.canvas.bind('<B1-Motion>', self.move_to)
        self.canvas.bind('<MouseWheel>', self.wheel)  # with Windows and MacOS, but not Linux
        self.canvas.bind('<Button-5>', self.wheel)  # only with Linux, wheel scroll down
        self.canvas.bind('<Button-4>', self.wheel)  # only with Linux, wheel scroll up

        # Text is used to set proper coordinates to the image. You can make it invisible.
        self.text = self.canvas.create_text(0, 0, anchor='nw')
        self.plot_test_rectangles()
        self.layout()
        self.show_image()
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))


        # Layout
    def layout(self):
#        self.master.rowconfigure(1, weight=1)
#        self.master.columnconfigure(0, weight=1)
#        self.master.columnconfigure(7, weight=1)

        self.label.grid(row=0, column=0, columnspan=5, sticky=W)
        self.label_ul_static_x.grid(row=1, column=0, sticky=W)
        self.label_ul_static_y.grid(row=2, column=0, sticky=W)
        self.label_ul_x.grid(row=1, column=2, sticky=W)
        self.label_ul_y.grid(row=2, column=2, sticky=W)

        self.entry_ul_x.grid(row=1, column=1, columnspan=1, sticky=W)
        self.entry_ul_y.grid(row=2, column=1, columnspan=1, sticky=W)

        self.button_ok.grid(row=1, column=3, rowspan=2)
        self.button_reset.grid(row=1, column=4, rowspan=2)
#        self.button_ok_ul_y.grid(row=2, column=3, sticky=W)
#        self.button_reset_ul_y.grid(row=2, column=4, sticky=W)

        self.canvas.grid(row=3, column=0, columnspan=6, sticky=W+E+N+S)

        # Make the canvas expandable
        self.master.rowconfigure(3, weight=1)
        self.master.columnconfigure(5, weight=1)

#    def load_img(self, array=None):
#        '''Loads image as numpy array into window'''
#        if array is None:
#            array = np.ones((640,2000))*150
#        img =  ImageTk.PhotoImage(image=Image.fromarray(array))
#
#        canvas = Canvas(root, width=300, height=300)
#        canvas.pack()
#        canvas.create_image(20,20, anchor="nw", image=img)

    def plot_test_rectangles(self):
        '''Plots 10 random test rectangles on the canvas'''
        self.imscale = 1.0
        self.imageid = None
        self.delta = 0.75
        width, height = self.image.size
        minsize, maxsize = 5, 20
        for n in range(10):
            x0 = random.randint(0, width - maxsize)
            y0 = random.randint(0, height - maxsize)
            x1 = x0 + random.randint(minsize, maxsize)
            y1 = y0 + random.randint(minsize, maxsize)
            color = ('red', 'orange', 'yellow', 'green', 'blue')[random.randint(0, 4)]
            self.canvas.create_rectangle(x0, y0, x1, y1, outline='black', fill=color,
                                         activefill='white', tags=n)

    def click(self, event):
        self.add_crosshairs(event)
        self.text_ul_x.set(self.ul_x)
        self.text_ul_y.set(self.ul_y)

    def add_crosshairs(self, event=None, width=1):
        if event is None:
            self.ul_x = int(self.entered_number_x / self.imscale)
            self.ul_y = int(self.entered_number_y / self.imscale)
        else:
            self.entered_number_x = int(event.x / self.imscale)
            self.entered_number_y = int(event.y / self.imscale)
            self.ul_x = self.entered_number_x
            self.ul_y = self.entered_number_y

        width, height = self.image.size
#        new_size = int(self.imscale * width), int(self.imscale * height)

        coords_x1 = (self.ul_x, 0, self.ul_x, height)
        coords_x2 = (self.ul_x+1, 0, self.ul_x+1, height)
        coords_y1 = (0, self.ul_y, width, self.ul_y)
        coords_y2 = (0, self.ul_y+1, width, self.ul_y+1)

        if self.canvas_id_x1 or self.canvas_id_y1:
            self.canvas.coords(self.canvas_id_x1, coords_x1)
            self.canvas.coords(self.canvas_id_x2, coords_x2)
            self.canvas.coords(self.canvas_id_y1, coords_y1)
            self.canvas.coords(self.canvas_id_y2, coords_y2)
        else:
            self.canvas_id_x1 = self.canvas.create_line(coords_x1, fill='red')
            self.canvas_id_x2 = self.canvas.create_line(coords_x2, fill='blue')
            self.canvas_id_y1 = self.canvas.create_line(coords_y1, fill='green')
            self.canvas_id_y2 = self.canvas.create_line(coords_y2, fill='white')
#        canvas_id_y = self.canvas.create_line(y, 0, y, 200, fill='red')
#        canvas_id_x = self.canvas.create_line(0, x, 200, x, fill='red')
#        self.canvas.create_rectangle(self.ul_y, self.ul_y+1, 0, 200,
#                                     outline='red', fill='white',
#                                     activefill='red', tags=n)
#        self.canvas.create_rectangle(0, 200, self.ul_x, self.ul_x + 1,
#                                     outline='red', fill='white',
#                                     activefill='red', tags=n)

    def validate(self, new_text, direction):
        if not new_text or not direction: # the field is being cleared
            self.entered_number_x = 0
            self.entered_number_y = 0
            self.add_crosshairs()
            return True
        if direction == 'x':
            try:
                self.entered_number_x = int(new_text)
                if (isinstance(self.entered_number_x, int) and
                    isinstance(self.entered_number_y, int)):
                    self.add_crosshairs()
                return True
            except ValueError:
                return False
        elif direction == 'y':
            try:
                self.entered_number_y = int(new_text)
                if (isinstance(self.entered_number_x, int) and
                    isinstance(self.entered_number_y, int)):
                    self.add_crosshairs()
                return True
            except ValueError:
                return False

    def update(self, method):
        if method == "OK":
            self.ul_x = self.entered_number_x
            self.ul_y = self.entered_number_y
#        elif method == "OK_y":
#            self.ul_y = self.entered_number
        elif method == "reset": # reset
            self.ul_x = 0
            self.ul_y = 0
#        elif method == "reset_y":
#            self.ul_y = 0

        self.text_ul_x.set(self.ul_x)
        self.text_ul_y.set(self.ul_y)
        self.entry_ul_x.delete(0, END)
        self.entry_ul_y.delete(0, END)

    def move_from(self, event):
        ''' Remember previous coordinates for scrolling with the mouse '''
        self.canvas.scan_mark(event.x, event.y)

    def move_to(self, event):
        ''' Drag (move) canvas to the new position '''
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def wheel(self, event):
        ''' Zoom with mouse wheel '''
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120:
            scale        *= self.delta
            self.imscale *= self.delta
        if event.num == 4 or event.delta == 120:
            scale        /= self.delta
            self.imscale /= self.delta
        # Rescale all canvas objects
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.canvas.scale('all', x, y, scale, scale)
        self.show_image()
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def show_image(self):
        ''' Show image on the Canvas '''
        if self.imageid:
            self.canvas.delete(self.imageid)
            self.imageid = None
            self.canvas.imagetk = None  # delete previous image from the canvas
        width, height = self.image.size
        new_size = int(self.imscale * width), int(self.imscale * height)
        imagetk = ImageTk.PhotoImage(self.image.resize(new_size))
        # Use self.text object to set proper coordinates
        self.imageid = self.canvas.create_image(self.canvas.coords(self.text),
                                                anchor='nw', image=imagetk)
        self.canvas.lower(self.imageid)  # set it into background
        self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection

root = Tk()
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (int(w*0.5), int(h*0.9)))
my_gui = GUI_image_crop(root)

root.mainloop()
