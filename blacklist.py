import os
from tkinter import Tk, Label, Button, StringVar
from PIL import Image, ImageTk


class BlacklistGUI:
    def __init__(self, master):
        self.master = master
        master.title('ATAI Blacklist')

        self.reject_button = Button(master, text='Reject', command=self.reject) \
            .grid(row=2, column=0, columnspan=2, sticky='W')

        self.accept_button = Button(master, text='Accept', command=self.accept) \
            .grid(row=2, column=2, columnspan=3, sticky='E')

        self.close_button = Button(master, text='Quit', command=master.quit) \
            .grid(row=3, columnspan=5)

        self.image_panels = [
            Label(root)
            for _ in range(10)
        ]

        self.status_text = StringVar()
        self.status = Label(master, textvariable=self.status_text, bd=1, relief='sunken', anchor='w')
        self.status.status_text = self.status_text
        self.status.grid(row=4, columnspan=5, sticky='news')

        self.filenames = [f.split('/')[-1] for f in os.listdir('notMNIST_large/A/')]

        self.filename_index = 0

        self.blacklist = set()

        for i, panel in enumerate(self.image_panels):
            panel.grid(row=i // 5, column=i % 5)

        self.update_panels()
        self.update_status()

        self.master.bind('<Return>', self.accept)
        self.master.bind('<space>', self.accept)
        self.master.bind('<BackSpace>', self.reject)
        self.master.bind('<Escape>', self.quit)

    def update_panels(self):
        filename = self.filenames[self.filename_index]

        for i, panel in enumerate(self.image_panels):
            try:
                image = ImageTk.PhotoImage(Image.open(f'notMNIST_large/{chr(65+i)}/{filename}'))
            except FileNotFoundError:
                image = None
            panel.configure(image=image)
            panel.image = image

    def update_status(self):
        self.status_text.set(f'{self.filename_index + 1} / {len(self.filenames)}')

    def advance(self):
        if self.filename_index < len(self.filenames) - 1:
            self.filename_index += 1

    def accept(self, event=None):
        self.advance()
        self.update_panels()
        self.update_status()

    def reject(self, event=None):
        filename = self.filenames[self.filename_index]
        self.blacklist.add(filename)

        self.advance()
        self.update_panels()
        self.update_status()

    def quit(self, event=None):
        with open('blacklist.txt', 'w') as f:
            for filename in self.blacklist:
                f.write(f'{filename}\n')

        self.master.quit()


if __name__ == '__main__':
    root = Tk()
    my_gui = BlacklistGUI(root)
    root.mainloop()
