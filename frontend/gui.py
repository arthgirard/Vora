import customtkinter as ctk

# ================= APP =================
class App(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.geometry("1280x720")
        self.title("Vora")

        ctk.set_appearance_mode("Light")
        ctk.set_widget_scaling(1.0)

        self.pages = {}
        self.buttons = {}
        self.active_page = "Démarrage"

        # Sidebar animée
        self.sidebar_width = 220
        self.sidebar_hidden_width = 10
        self.sidebar_current_width = self.sidebar_hidden_width
        self.sidebar_anim_speed = 20

        self.create_layout()
        self.create_pages()
        self.create_sidebar_buttons()
        self.show_page(self.active_page)

    # ================= LAYOUT =================
    def create_layout(self):
        self.sidebar = ctk.CTkFrame(self, width=self.sidebar_hidden_width, corner_radius=10)
        self.sidebar.pack(side="left", fill="y")

        ctk.CTkLabel(self.sidebar, text="MENU", font=("Manrope", 25, "bold")).pack(pady=20)

        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(side="right", expand=True, fill="both")

    # ================= SIDEBAR ANIMATION =================
    def expand_sidebar(self, event=None):
        if self.sidebar_current_width < self.sidebar_width:
            self.sidebar_current_width += self.sidebar_anim_speed
            if self.sidebar_current_width > self.sidebar_width:
                self.sidebar_current_width = self.sidebar_width
            self.sidebar.configure(width=self.sidebar_current_width)
            self.after(10, self.expand_sidebar)

    # ================= PAGES =================
    def create_pages(self):
        for name in ["Démarrage", "Analyse", "Paramètres"]:
            frame = ctk.CTkFrame(self.main_container)
            ctk.CTkLabel(frame, text=f"Page {name}", font=("Manrope", 28, "bold")).pack(pady=40)
            self.pages[name] = frame

        # Bouton test Démarrage
        ctk.CTkButton(
            self.pages["Démarrage"],
            text="Bouton Démarrage",
            font=("Manrope", 14, "bold"),
            command=self.start_application
        ).pack(pady=20)

        # ===== CONTENU PARAMETRES =====
        ctk.CTkLabel(self.pages["Paramètres"], text="Thème", font=("Manrope", 16, "bold")).pack(pady=10)

        self.light_btn = ctk.CTkButton(
            self.pages["Paramètres"],
            text="Light Mode",
            command=lambda: self.set_theme("Light")
        )
        self.light_btn.pack(pady=5)

        self.dark_btn = ctk.CTkButton(
            self.pages["Paramètres"],
            text="Dark Mode",
            command=lambda: self.set_theme("Dark")
        )
        self.dark_btn.pack(pady=5)

        ctk.CTkLabel(self.pages["Paramètres"], text="UI Scaling", font=("Manrope", 16, "bold")).pack(pady=10)

        scaling_option = ctk.CTkOptionMenu(
            self.pages["Paramètres"],
            values=["80%", "90%", "100%", "110%", "120%"],
            command=self.change_scaling
        )
        scaling_option.set("100%")
        scaling_option.pack(pady=5)

    # ================= SIDEBAR BUTTONS =================
    def create_sidebar_buttons(self):
        for name in self.pages:
            btn = ctk.CTkButton(
                self.sidebar,
                text=name,
                font=("Manrope", 14, "bold"),
                command=lambda n=name: self.change_page(n)
            )
            btn.pack(pady=10, padx=20, fill="x")
            self.buttons[name] = btn

        # ===== QUITTER =====
        self.quit_btn = ctk.CTkButton(
            self.sidebar,
            text="Quitter",
            font=("Manrope", 14, "bold"),
            command=self.confirm_quit
        )
        self.quit_btn.pack(pady=(40, 20), padx=20, fill="x")

    # ================= PAGE MANAGEMENT =================
    def change_page(self, name):
        self.show_page(name)

    def show_page(self, name):
        self.active_page = name
        for page in self.pages.values():
            page.pack_forget()
        self.pages[name].pack(expand=True, fill="both")
        self.apply_colors()

    # ================= COLORS =================
    def update_colors(self):
        if ctk.get_appearance_mode() == "Dark":
            self.sidebar.configure(fg_color="#1A1A2E")
            return {"normal": "#3399FF", "hover": "#66B2FF", "active": "#003F7F", "text": "white"}
        else:
            self.sidebar.configure(fg_color="#F0F0F0")
            return {"normal": "#1A8CFF", "hover": "#4DA6FF", "active": "#005EA6", "text": "white"}

    def apply_colors(self):
        colors = self.update_colors()
        for name, btn in self.buttons.items():
            if name == self.active_page:
                btn.configure(fg_color=colors["active"], hover_color=colors["hover"], text_color=colors["text"])
            else:
                btn.configure(fg_color=colors["normal"], hover_color=colors["hover"], text_color=colors["text"])
        self.quit_btn.configure(fg_color=colors["normal"], hover_color=colors["hover"], text_color=colors["text"])

    # ================= SETTINGS =================
    def set_theme(self, mode):
        ctk.set_appearance_mode(mode)
        self.apply_colors()

    def change_scaling(self, value):
        scaling = int(value.replace("%", "")) / 100
        ctk.set_widget_scaling(scaling)

    # ================= QUIT =================
    def confirm_quit(self):
        self.quit()

    # ================= DEMARRAGE =================
    def start_application(self):
        print("Démarrage de l'application lancé !")


# ================= START =================
if __name__ == "__main__":
    app = App()
    app.mainloop()
