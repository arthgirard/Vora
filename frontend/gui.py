import customtkinter as ctk

# ================= CONFIGURATION OPTIMISÉE =================
COLORS = {
    "app_bg": ("#F3F3F3", "#202020"), 
    "container_bg": ("#FFFFFF", "#1C1C1C"),
    "hover_bg": ("#E5E5E5", "#2D2D2D"),
    "border_color": ("#E0E0E0", "#2D2D2D"),
    "active_bg": ("#EBEBEB", "#323232"),
    "accent": "#0078D4",
    "text": ("#000000", "#FFFFFF"),
    "switch_bg": ("#E0E0E0", "#000000"),
}

class Fonts:
    @classmethod
    def init(cls):
        cls.MAIN = ctk.CTkFont(family="Segoe UI Variable", size=13)
        cls.BOLD = ctk.CTkFont(family="Segoe UI Variable", size=13, weight="bold")
        cls.TITLE = ctk.CTkFont(family="Segoe UI Variable", size=18, weight="bold")
        cls.TOPBAR = ctk.CTkFont(family="Segoe UI Variable", size=18, weight="bold")
        cls.SMALL_BOLD = ctk.CTkFont(family="Segoe UI Variable", size=10, weight="bold")

# ================= WIDGETS =================

class SidebarItem(ctk.CTkFrame):
    def __init__(self, master, text, command):
        super().__init__(master, fg_color="transparent", corner_radius=6, cursor="hand2")
        self.command = command
        self._active = False

        self.indicator = ctk.CTkFrame(self, width=4, height=20, corner_radius=2, fg_color="transparent")
        self.indicator.pack(side="left", padx=(0, 15), pady=12)

        self.label = ctk.CTkLabel(self, text=text, font=Fonts.MAIN, text_color=COLORS["text"], anchor="w")
        self.label.pack(side="left", fill="x", expand=True)

        for w in [self, self.label, self.indicator]:
            w.bind("<Enter>", self._on_enter)
            widget_callback = lambda e: self.command()
            w.bind("<Button-1>", widget_callback)
            w.bind("<Leave>", self._on_leave)

    def set_active(self, active):
        self._active = active
        self.configure(fg_color=COLORS["active_bg"] if active else "transparent")
        self.indicator.configure(fg_color=COLORS["accent"] if active else "transparent")
        self.label.configure(font=Fonts.BOLD if active else Fonts.MAIN)

    def _on_enter(self, e):
        if not self._active: self.configure(fg_color=COLORS["hover_bg"])
    def _on_leave(self, e):
        if not self._active: self.configure(fg_color="transparent")

class ThemeToggle(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color=COLORS["switch_bg"], corner_radius=15, height=30, width=90)
        self.state = ctk.get_appearance_mode()
        
        self.label = ctk.CTkLabel(self, text="", font=Fonts.SMALL_BOLD, text_color=("black", "white"))
        self.knob = ctk.CTkFrame(self, width=22, height=22, corner_radius=11, fg_color="white")
        
        self.update_ui()
        for w in [self, self.label, self.knob]:
            w.bind("<Button-1>", self.toggle)

    def update_ui(self):
        if self.state == "Light":
            self.label.configure(text="Sombre") 
            self.label.place(relx=0.42, rely=0.5, anchor="center")
            self.knob.place(relx=0.82, rely=0.5, anchor="center")
        else:
            self.label.configure(text="Clair") 
            self.label.place(relx=0.58, rely=0.5, anchor="center")
            self.knob.place(relx=0.18, rely=0.5, anchor="center")

    def toggle(self, e=None):
        self.state = "Dark" if self.state == "Light" else "Light"
        ctk.set_appearance_mode(self.state)
        self.update_ui()

# ================= APPLICATION PRINCIPALE =================

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        Fonts.init()

        self.geometry("960x540")
        self.title("Vora")
        self.configure(fg_color=COLORS["app_bg"])

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.pages = {}
        self.sidebar_buttons = {}
        self.current_page = None

        self._setup_ui()
        self.show_page("Démarrage")

    def _setup_ui(self):
        sidebar_cont = ctk.CTkFrame(self, fg_color="transparent")
        sidebar_cont.grid(row=0, column=0, sticky="nsew", padx=(10, 0), pady=10)
        
        self.sidebar = ctk.CTkFrame(sidebar_cont, width=220, corner_radius=12, fg_color=COLORS["app_bg"])
        self.sidebar.pack(fill="both", expand=True)
        self.sidebar.pack_propagate(False)

        ctk.CTkLabel(self.sidebar, text="≡  Vora", font=Fonts.TITLE, anchor="w").pack(fill="x", padx=20, pady=(20, 25))

        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=0, column=1, sticky="nsew", padx=15, pady=10)

        for name in ["Démarrage", "Analyse", "Paramètres"]:
            btn = SidebarItem(self.sidebar, name, lambda n=name: self.show_page(n))
            btn.pack(fill="x", pady=2, padx=10)
            self.sidebar_buttons[name] = btn
            
            page_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
            self._build_page_structure(page_frame, name)
            self.pages[name] = page_frame

        ThemeToggle(self.sidebar).pack(side="bottom", pady=25, padx=20, anchor="w")

    def _build_page_structure(self, parent, name):
        topbar = ctk.CTkFrame(parent, fg_color=COLORS["container_bg"], height=65, corner_radius=10, border_width=1, border_color=COLORS["border_color"])
        topbar.pack(fill="x", pady=(0, 10))
        topbar.pack_propagate(False)

        ctk.CTkLabel(topbar, text=name, font=Fonts.TOPBAR, text_color=COLORS["text"]).pack(side="left", padx=20)

        if name == "Démarrage":
            self.run_btn = ctk.CTkButton(
                topbar, 
                text="+ Exécuter une tâche", 
                font=Fonts.BOLD, 
                height=32, 
                fg_color="transparent", 
                border_width=1, 
                border_color=COLORS["border_color"],
                text_color=COLORS["text"], 
                hover_color=COLORS["hover_bg"], 
                command=self.run_backend_logic
            )
            self.run_btn.pack(side="right", padx=20)

        ctk.CTkFrame(parent, fg_color=COLORS["container_bg"], corner_radius=10, 
                     border_width=1, border_color=COLORS["border_color"]).pack(fill="both", expand=True)

    def show_page(self, name):
        if self.current_page == name: return

        if self.current_page:
            self.pages[self.current_page].pack_forget()
            self.sidebar_buttons[self.current_page].set_active(False)

        self.pages[name].pack(fill="both", expand=True)
        self.sidebar_buttons[name].set_active(True)
        self.current_page = name

    def run_backend_logic(self):
        print("Démarrage")

if __name__ == "__main__":
    app = App()
    app.mainloop()