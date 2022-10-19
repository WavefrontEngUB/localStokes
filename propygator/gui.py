#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from .fields import HighNAField
from .Misc import zernike_p
from matplotlib import rcParams
rcParams["figure.autolayout"] = True
from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg)
from matplotlib.figure import Figure
import tkinter as tk
import tkinter.ttk as ttk
import sys

class FocalFieldViewer:
    """Program to calculate, visuzlize and explore the 3D distribution of
    the electromagnetic field a the focal plane of a high numerical aperture
    optical system."""
    def __init__(self, master):
        # Initializing local variables
        self.EntranceField = None   # Initializing to None the object field
        self.master = master
        self.master.st = ttk.Style()
        self.master.st.theme_use("clam") # TODO: disable it for windows users
        self.master.title("Focal Field Viewer")
        self.lamb = 628e-6
        self.f = 5

        # Initializing the interface
        # TODO: Add zernike polynomials
        # Notebook container
        self.notebook = ttk.Notebook(master)
        # Optical constants
        self.ConstantWindow = ConstantWindow(master, cmd=self.compute,
                text="Physical constants")
        # Zernike coefficients
        self.ZernikeWindow = ZernikeWindow(master)
        #self.ConfigWindow.pack(anchor="e",side=tk.LEFT, fill="y", 
        #                    padx=5, expand=False)
        self.notebook.add(self.ConstantWindow, text="Optical constants")
        self.notebook.pack(anchor="e", side=tk.LEFT, fill="y", padx=5,
                expand=False)

        self.notebook.add(self.ZernikeWindow, text="Zernike p.")
        self.notebook.pack(anchor="e", side=tk.LEFT, fill="y", padx=5,
                expand=False)

        #self.sep = ttk.Separator(master, orient="vertical")
        #self.sep.pack(fill="y", side=tk.LEFT, expand=False)

        self.PlotWindow = PlotWindow(master,)
        self.PlotWindow.pack(anchor="w", side=tk.RIGHT, fill="both",
                expand=True)

        # Convenient key bindings
        self.master.bind("q", self.quit)
        self.master.bind("<Return>", self.compute)

    def compute(self, event=None):
        # Get all physical constants
        fill_f, NA, Lf, zr, pol, L_pc = self.ConstantWindow.dump()
        # Get zernike values
        zernikes, indices = self.ZernikeWindow.dump()
        ind_calculate = {}      # Indexs to calculate
        for i, val in enumerate(zernikes):
            if abs(val) > 0:
                ind_calculate[i] = val
        ## Create appropriate field
        nx = 255
        ny = nx
        # Calculating the sampling region so that the focal size is Lx
        Lx = nx * self.f / 4 / Lf
        Ly = Lx
        #lamb *= 1e-6 # mm
        # Field per se
        Ein = np.zeros((ny, nx, 2), dtype=np.complex128)
        x = np.linspace(-Lx, Lx, nx)
        y = np.linspace(-Ly, Ly, ny)
        x, y = np.meshgrid(x, y)

        # Size of the E.P.
        d = self.f * np.tan(np.arcsin(NA))
        r2 = x * x + y * y
        sigma2 = fill_f * fill_f * d * d
        A = np.exp(-r2 / sigma2).astype(np.complex128)
        # Calculating mu
        L_pc = d * L_pc
        if L_pc != 0:
            mu = np.exp(-(x*x+y*y)/L_pc/L_pc)
        else:
            mu = None
        # Adding pupil function to the amplitude. Represented
        # by the Zernike polynomials.
        # TODO: Add Pupil function for light needles!
        """
        if ind_calculate:
            r_norm = np.sqrt(r2)/d  # Normalized coordinates. 
            phi = np.arctan2(y, x)
            P = np.ones((nx, ny), dtype=np.complex128)
            W = np.empty((nx, ny), dtype=np.float64)
            for coeffs, c in ind_calculate:
                W[:, :] = c*zernike_p(r_norm, phi, *coeffs)
                P[:, :] = P*np.exp(2j*W/self.lamb)
            A[:] = A*P
        """

        # Caclulation of the input field
        if pol == "Linear X":
            Ein[:, :, 0] = A
        elif pol == "Linear Y":
            Ein[:, :, 1] = A
        elif pol == "Radial":
            phi = np.arctan2(y, x)
            Ein[:, :, 0] = A * np.cos(phi)
            Ein[:, :, 1] = A * np.sin(phi)
        elif pol == "Star":
            phi = np.arctan2(y, x)
            Ein[:, :, 0] = A * np.cos(4*phi)
            Ein[:, :, 1] = A * np.sin(4*phi)
            
        self.EntranceField = HighNAField(Ein, self.f, NA, (Lx, Ly), 
                self.lamb, n=1.)
        if ind_calculate:
            self.EntranceField.construct_aberration(ind_calculate, 
                    order=indices)

        self.FocalField = self.EntranceField.propagate(0, mu)

        # Transverse intensity
        nz = 64
        I_trans = np.empty((nx, ny, 3, nz), dtype=np.float64)
        prop = self.EntranceField.propagate
        dims = zr, Lf
        z = np.linspace(-dims[0]*self.lamb, dims[0]*self.lamb, nz)
        i = 0   # Variable to count in the z_loop function
        self.z_loop(prop, I_trans, z, mu, i, nz, dims)

        self.PlotWindow.plot_section(self.FocalField, Lf)

    def z_loop(self, prop, I_trans, z, mu, i, nz, dims):
        if i < nz:
            I = prop(z[i], mu)
            #I = np.sum(I, axis=(1, 2))
            I_trans[:, :, :, i] = np.real(I[:])
            i += 1
            self.master.after(0, self.z_loop, prop, I_trans, z, mu, i, nz, dims)
        else:   # If finished, plot the results
            self.PlotWindow.plot_z(I_trans, dims)

    def quit(self, event):
        self.master.destroy()
        sys.exit(0)

class ConstantWindow(ttk.Frame): #ttk.LabelFrame if text
    def __init__(self, master, cmd, text=""):
        #ttk.LabelFrame.__init__(self, master, text=text)
        ttk.Frame.__init__(self, master)

        # Fill factor of the gaussian beam
        self.fill_f_name = ttk.Label(self, text="Fill factor")
        self.fill_f_name.grid(row=0, column=0, columnspan=2, sticky=tk.W)

        self.fill_f_entry = ttk.Entry(self, width=10)
        self.fill_f_entry.delete(0, tk.END)
        self.fill_f_entry.insert(0, "1")
        self.fill_f_entry.grid(row=1, column=0, sticky=tk.W)
        
        # Focal length text entry
        #self.f_name = ttk.Label(self, text="Focal length")
        #self.f_name.grid(row=2, column=0, columnspan=2, sticky=tk.W)

        #self.f_entry = ttk.Entry(self, width=10)
        #self.f_entry.delete(0, tk.END)
        #self.f_entry.insert(0, "5")
        #self.f_entry.grid(row=3, column=0, sticky=tk.W)

        #self.f_units = ttk.Label(self, text="mm")
        #self.f_units.grid(row=3, column=1, sticky=tk.W)

        # Numerical aperture entry
        self.NA_name = ttk.Label(self, text="Numerical aperture")
        self.NA_name.grid(row=4, column=0, columnspan=2, sticky=tk.W)

        self.NA_entry = ttk.Entry(self, width=10)
        self.NA_entry.delete(0, tk.END)
        self.NA_entry.insert(0, "0.75")
        self.NA_entry.grid(row=5, column=0, sticky=tk.W)

        # Half length text entry
        self.Lx_name = ttk.Label(self, text="Half size focal window")
        self.Lx_name.grid(row=6, column=0, columnspan=2, sticky=tk.W)

        self.Lx_entry = ttk.Entry(self, width=10)
        self.Lx_entry.delete(0, tk.END)
        self.Lx_entry.insert(0, "8")
        self.Lx_entry.grid(row=7, column=0, sticky=tk.W)

        self.Lx_units = ttk.Label(self, text="λ")
        self.Lx_units.grid(row=7, column=1, sticky=tk.W)

        # Z distances calculation
        self.z_name = ttk.Label(self, text="Half z range")
        self.z_name.grid(row=8, column=0, columnspan=2, sticky=tk.W)
        
        self.z_entry = ttk.Entry(self, width=10)
        self.z_entry.delete(0, tk.END)
        self.z_entry.insert(0, 8)
        self.z_entry.grid(row=9, column=0, sticky=tk.W)

        # Partial coherence
        self.pc_entry = ttk.Entry(self, width=10)
        self.pc_entry.delete(0, tk.END)
        self.pc_entry.insert(0, "0.5")
        self.pc_entry.configure(state="disabled")
        self.pc_entry.grid(row=11, column=0, sticky=tk.W)

        self.var_pc = tk.IntVar()
        self.partcoh_b = ttk.Checkbutton(self, text="Partial coherence",
                variable=self.var_pc, command=self.pc_check)
        self.partcoh_b.var = self.var_pc
        self.partcoh_b.grid(row=10, column=0, sticky=tk.W)

        # Wavelength of the light
        #self.wl_name = ttk.Label(self, text="Wavelength")
        #self.wl_name.grid(row=10, column=0, columnspan=2, sticky=tk.W)

        #self.wl_entry = ttk.Entry(self, width=10)
        #self.wl_entry.delete(0, tk.END)
        #self.wl_entry.insert(0, "628")
        #self.wl_entry.grid(row=11, column=0, sticky=tk.W)

        #self.wl_units = ttk.Label(self, text="nm")
        #self.wl_units.grid(row=11, column=1, sticky=tk.W)

        # Combobox, selection of polarization
        self.pol_name = ttk.Label(self, text="Polarization")
        self.pol_name.grid(row=12, column=0, columnspan=2,
                sticky=tk.W)

        self.pol_box = ttk.Combobox(self)
        self.pol_box["values"] = ("Linear X", "Linear Y", "Radial", "Star")
        self.pol_box.current(0)
        self.pol_box["state"] = "readonly"
        self.pol_box.grid(row=13, column=0, columnspan=2)

        # Compute button
        self.comp_b = ttk.Button(self, text="Compute", command=cmd)
        self.comp_b.grid(row=14, column=0, sticky=tk.W+tk.S,
                pady=20)

    def dump(self):
            fill_f = float(self.fill_f_entry.get())
            #f      = float(self.f_entry.get())
            NA     = float(self.NA_entry.get())
            Lx     = float(self.Lx_entry.get())
            zr     = float(self.z_entry.get())
            #lamb   = float(self.wl_entry.get())
            pol    =       self.pol_box.get()
            pc = self.var_pc.get()
            if pc == 1:
                L_pc = float(self.pc_entry.get())
            else:
                L_pc = 0
            return (fill_f, NA, Lx, zr, pol, L_pc)

    def pc_check(self):
        pc = self.var_pc.get()
        if pc == 0:
            self.pc_entry.configure(state="disabled")
        else:
            self.pc_entry.configure(state="normal")

class ZernikeWindow(ttk.Frame):
    """Window to contain the values of Zernike Polynomials."""
    def __init__(self, master):
        # TODO: Expand the class and add return methods. Dump all coeff.
        ttk.Frame.__init__(self, master, )
        self.master=master
        self.labels = []        # Container for the labels
        self.zk_entries = []    # Container for the entries
        self.zk_values = [0.]*20
        # Standard representations of Zernike polynomial indices
        """
        self.OSA    = {0:(0, 0), 1:(1, -1), 2:(1,1), 3:(2,-2), 4:(2, 0),
                5:(2, 2), 6:(3, -3), 7:(3, -1), 8:(3, 1), 9:(3, 3), 10:(4, -4),
                11:(4, -2), 12:(4, 0), 13:(4, 2), 14:(4, 4), 15:(5, -5),
                16:(5, -3), 17:(5, -1), 18:(5, 1), 19:(5, 3)}
        # TODO: Change numeration in labels to reflect that Noll and 
        # Fringe start at 1
        self.Noll   = {0:(0, 0), 2:(1, -1), 1:(1,1), 4:(2,-2), 3:(2, 0),
                5:(2, 2), 8:(3, -3), 6:(3, -1), 7:(3, 1), 9:(3, 3), 14:(4, -4),
                12:(4, -2), 10:(4, 0), 11:(4, 2), 13:(4, 4), 19:(5, 5),
                18:(5, -3), 16:(5, -1), 15:(5, 1), 17:(5, 3)}
        self.Fringe = {0:(0, 0), 1:(1, 1), 2:(1, -1), 3:(2, 0), 4:(2, 2),
                5:(2, -2), 6:(3, 1), 7:(3, -1), 8:(4, 0), 9:(3, 3), 10:(3, -3),
                11:(4, 2), 12:(4, -2), 13:(5, 1), 14:(5, -1), 15:(6, 0), 
                16:(4, 4), 17:(4, -4), 18:(5, 3), 19:(5, -3)}
        self.keys = {"OSA":self.OSA, "Noll":self.Noll, "Fringe":self.Fringe}
        """

        # Menu to select the Zernike polynomial convention
        self.label_c = ttk.Label(self, text="Index representation")
        self.label_c.grid(row=0, column=0, sticky=tk.W, columnspan=2)

        self.menu_c = ttk.Combobox(self)
        self.menu_c["values"] = ["OSA", "Noll", "Fringe"]
        self.menu_c.current(0)
        self.menu_c["state"] = "readonly"
        self.menu_c.grid(row=1, column=0, columnspan=2)
        
        # Creating the entry boxes for the Zernike polynomials
        ncol = 2
        nrow = 10
        pdx = 10
        append_l = self.labels.append
        append_z = self.zk_entries.append
        for i in range(0, 2*nrow, 2):
            for j in range(ncol):
                n = i*ncol//2+j
                text = "Z {:d}".format(n)
                append_l(ttk.Label(self, text=text))
                self.labels[n].grid(row=(i*nrow)+2, column=j, padx=pdx)
                append_z(ttk.Entry(self, width=8))
                self.zk_entries[n].delete(0, tk.END)
                self.zk_entries[n].insert(0, "0")
                self.zk_entries[n].grid(row=((i+1)*nrow+2), column=j, padx=pdx)

    def dump(self):
        """Dump all info about Zernike polynomial coefficients"""
        # Dump index representation
        rep = self.menu_c.get()
        for i, entry in enumerate(self.zk_entries):
            self.zk_values[i] = float(entry.get())
        return self.zk_values, rep

class PlotWindow(ttk.Notebook):
    """Tkinter Notebook to hold all the plot data."""
    # TODO: Change into notebook for several different representations inside
    # the same window...
    def __init__(self, master, text=""):
        ttk.Notebook.__init__(self, master)
        self.cb1 = None
        self.cbz = None
        self.cyz = None
        self.bg = "#dcdad5"
        
        # Plot of intensities per component, first notebook page.
        self.fig1 = Figure(figsize=(6.4, 4.8), dpi=100)
        #self.fig1.patch.set_facecolor(master["bg"])     # Uniform looking bg
        self.fig1.patch.set_facecolor(self.bg)     # Uniform looking bg
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self)
        self.ax1 = self.fig1.add_subplot(221)
        self.ax1.set_title(r"$\vert E_x \vert^2$")
        self.ax2 = self.fig1.add_subplot(222)
        self.ax2.set_title(r"$\vert E_y \vert^2$")
        self.ax3 = self.fig1.add_subplot(223)
        self.ax3.set_title(r"$\vert E_z \vert^2$")
        self.ax4 = self.fig1.add_subplot(224)
        self.ax4.set_title(r"$\vert E_T \vert^2$")
        self.canvas1.draw()
        self.add(self.canvas1.get_tk_widget(), text="X-Y Intensities")
        # Init imshow plots!
        ext=[-1, 1, -1, 1]
        I = np.zeros((256, 256), dtype=np.float64)
        self.c1 = self.ax1.imshow(I, cmap="inferno", extent=ext)
        self.ax1.set_title(r"$\vert E_x \vert^2$")
        self.c2 = self.ax2.imshow(I, cmap="inferno", extent=ext)
        self.ax2.set_title(r"$\vert E_y \vert^2$")
        self.c3 = self.ax3.imshow(I, cmap="inferno", extent=ext)
        self.ax3.set_title(r"$\vert E_z \vert^2$")
        self.c4 = self.ax4.imshow(I, cmap="inferno", extent=ext)
        self.ax4.set_title(r"$\vert E_T \vert^2$")
        # Colorbars
        self.cb1 = self.fig1.colorbar(self.c1, ax=self.ax1)
        self.cb2 = self.fig1.colorbar(self.c2, ax=self.ax2)
        self.cb3 = self.fig1.colorbar(self.c3, ax=self.ax3)
        self.cb4 = self.fig1.colorbar(self.c4, ax=self.ax4)


        # Adding a plot of the transverse intensity in x-z plane,
        # second notebook page.
        Iz = np.zeros((256, 64), dtype=np.float64)
        dims = [-1, 1, -1, 1]
        self.fig2 = Figure(figsize=(6.4, 4.8), dpi=100)
        self.fig2.patch.set_facecolor(self.bg)     # Uniform looking bg
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self)
        self.axz = self.fig2.add_subplot(111)
        self.axz.set_title(r"$\int\vert E(x, y, z)\vert^2 dy$")
        self.cz = self.axz.imshow(Iz, aspect="auto", cmap="inferno",
                extent=[-dims[1], dims[1], -dims[0], dims[0]])
        self.cbz = self.fig2.colorbar(self.cz, ax=self.axz)
        self.canvas2.draw()
        self.add(self.canvas2.get_tk_widget(), text="X-Z Intensity")
        #self.canvas2.get_tk_widget().pack(side=tk.BOTTOM, fill="both", expand=True)

        # Adding a plot of the transverse intensity in y-z plane,
        # second notebook page.
        self.fig3 = Figure(figsize=(6.4, 4.8), dpi=100)
        self.fig3.patch.set_facecolor(self.bg)     # Uniform looking bg
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=self)
        self.ayz = self.fig3.add_subplot(111)
        self.ayz.set_title(r"$\int\vert E(x, y, z)\vert^2 dx$")
        self.cyz = self.ayz.imshow(Iz, aspect="auto", cmap="inferno",
                extent=[-dims[1], dims[1], -dims[0], dims[0]])
        self.cby = self.fig2.colorbar(self.cyz, ax=self.ayz)
        self.canvas3.draw()
        self.add(self.canvas3.get_tk_widget(), text="Y-Z Intensity")

    def plot_section(self, Ein, Lx):
        #If = np.real(np.conj(Ein) * Ein)
        If = Ein/Ein.max()
        l = Lx
        ext = [-l, l, -l, l]

        # New intensities and max values
        Ix = If[:, :, 0]
        xmin, xmax = Ix.min(), Ix.max()
        Iy = If[:, :, 1]
        ymin, ymax = Iy.min(), Iy.max()
        Iz = If[:, :, 2]
        zmin, zmax = Iz.min(), Iz.max()
        It = np.sum(If, axis=2)
        tmax, tmin = It.max(), It.min()

        # Actualizing the data of the plots
        self.c1.set_data(Ix)
        self.c1.set_cmap(cmap="inferno")
        self.c1.set_extent(extent=ext)
        self.c1.set_clim(xmin, xmax)
        self.cb1.mappable.set_clim(xmin, xmax)

        self.c2.set_data(Iy)
        self.c2.set_cmap(cmap="inferno")
        self.c2.set_extent(extent=ext)
        self.c2.set_clim(ymin, ymax)
        self.cb2.mappable.set_clim(ymin, ymax)

        self.c3.set_data(Iz)
        self.c3.set_cmap(cmap="inferno")
        self.c3.set_extent(extent=ext)
        self.c3.set_clim(zmin, zmax)
        self.cb3.mappable.set_clim(zmin, zmax)

        self.c4.set_data(np.sum(If, axis=2))
        self.c4.set_cmap(cmap="inferno")
        self.c4.set_extent(extent=ext)
        self.c4.set_clim(tmin, tmax)
        self.cb4.mappable.set_clim(tmin, tmax)

        # Drawing
        self.canvas1.draw()

    def plot_z(self, Iz, dims):
        """Plot the transverse intensity along both axes."""
        # TODO: Add a z selector to select the plane of visualization in the
        # XY plane...
        nx, ny, _, nz = Iz.shape
        Izs = np.sum(Iz, axis=(2))
        Ixz = np.sum(Izs, axis=1)/Izs.max()
        xzmin, xzmax = Ixz.min(), Ixz.max()
        # Clenup
        # Plotting
        self.cz.set_data(Ixz) #aspect="auto", 
        self.cz.set_extent([-dims[0], dims[0], -dims[1], dims[1]])
        self.cz.set_clim(xzmin, xzmax)
        self.cbz.mappable.set_clim(xzmin, xzmax)
        self.canvas2.draw()

        # Clenup
        # TODO: Check if the representation is, in any way, correct.
        Iyz = np.sum(Izs, axis=0)/Izs.max()
        yzmin, yzmax = Iyz.min(), Iyz.max()
        # Plotting
        self.cyz.set_data(Iyz) #aspect="auto", cmap="inferno",
        self.cyz.set_extent(extent=[-dims[0], dims[0], -dims[1], dims[1]])
        self.cyz.set_clim(yzmin, yzmax)
        self.cby.mappable.set_clim(yzmin, yzmax)
        self.canvas3.draw()
