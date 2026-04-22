import flet as ft
import threading
import sys
import os

# ajout du chemin du backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
try:
    from blink_counter import BlinkCounter
except ImportError:
    BlinkCounter = None

COLORS = {
    "app_bg": {ft.ThemeMode.LIGHT: "#F3F3F3", ft.ThemeMode.DARK: "#202020"},
    "container_bg": {ft.ThemeMode.LIGHT: "#FFFFFF", ft.ThemeMode.DARK: "#1C1C1C"},
    "hover_bg": {ft.ThemeMode.LIGHT: "#E5E5E5", ft.ThemeMode.DARK: "#2D2D2D"},
    "border_color": {ft.ThemeMode.LIGHT: "#E0E0E0", ft.ThemeMode.DARK: "#2D2D2D"},
    "active_bg": {ft.ThemeMode.LIGHT: "#EBEBEB", ft.ThemeMode.DARK: "#323232"},
    "accent": "#0078D4",
    "text": {ft.ThemeMode.LIGHT: "#000000", ft.ThemeMode.DARK: "#FFFFFF"},
    "switch_bg": {ft.ThemeMode.LIGHT: "#E0E0E0", ft.ThemeMode.DARK: "#000000"},
}

def get_color(key, mode):
    val = COLORS.get(key)
    if isinstance(val, dict):
        return val[mode]
    return val

def create_border(width, color):
    # creation de bordure securisee pour eviter les modules deprecies
    side = ft.BorderSide(width, color)
    return ft.Border(top=side, bottom=side, left=side, right=side)

class SidebarItem(ft.Container):
    def __init__(self, text, on_click, is_active=False, mode=ft.ThemeMode.LIGHT):
        super().__init__()
        self.text_val = text
        self.on_click_action = on_click
        self.is_active = is_active
        self.mode = mode
        
        self.border_radius = 6
        self.padding = 8
        self.on_hover = self._hover
        self.on_click = self._click
        
        self.indicator = ft.Container(
            width=4, 
            height=20, 
            border_radius=2, 
            bgcolor=get_color("accent", self.mode) if is_active else ft.Colors.TRANSPARENT
        )
        
        self.label = ft.Text(
            value=text, 
            color=get_color("text", self.mode),
            size=13,
            weight=ft.FontWeight.BOLD if is_active else ft.FontWeight.NORMAL
        )
        
        self.content = ft.Row([self.indicator, self.label], alignment=ft.MainAxisAlignment.START)
        self.bgcolor = get_color("active_bg", self.mode) if is_active else ft.Colors.TRANSPARENT

    def _hover(self, e):
        # gestion du survol
        if not self.is_active:
            self.bgcolor = get_color("hover_bg", self.mode) if e.data == "true" else ft.Colors.TRANSPARENT
            self.update()

    def _click(self, e):
        # gestion du clic
        self.on_click_action(self.text_val)

    def set_active(self, active, mode):
        # mise a jour de l'etat actif
        self.is_active = active
        self.mode = mode
        self.bgcolor = get_color("active_bg", self.mode) if active else ft.Colors.TRANSPARENT
        self.indicator.bgcolor = get_color("accent", self.mode) if active else ft.Colors.TRANSPARENT
        self.label.weight = ft.FontWeight.BOLD if active else ft.FontWeight.NORMAL
        self.label.color = get_color("text", self.mode)
        self.update()

class App:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "Vora"
        self.page.window.width = 960
        self.page.window.height = 540
        self.page.padding = 15
        self.page.theme_mode = ft.ThemeMode.LIGHT
        self.page.bgcolor = get_color("app_bg", self.page.theme_mode)
        self.page.fonts = {"Segoe UI Variable": "Segoe UI Variable"}
        self.page.theme = ft.Theme(font_family="Segoe UI Variable")
        
        self.current_page = "Démarrage"
        self.sidebar_items = {}
        
        self.backend_thread = None
        self.backend_instance = None

        self.build_ui()

    def build_ui(self):
        # bascule de theme
        self.theme_btn = ft.Container(
            content=ft.Text("Sombre", size=10, weight="bold", color=get_color("text", self.page.theme_mode)),
            bgcolor=get_color("switch_bg", self.page.theme_mode),
            border_radius=15,
            width=90,
            height=30,
            alignment=ft.Alignment(0, 0),
            on_click=self.toggle_theme
        )

        # création du logo avec une référence pour changer la couleur plus tard
        self.logo_img = ft.Image(
                src = "logo_black.png",
                height = 40,
                fit="contain",
                )

        # barre laterale
        sidebar_content = ft.Column(
            controls=[
                ft.Container(
                   content=self.logo_img,
                   padding=10
                )
            ],
            expand=True
        )
        
        for name in ["Démarrage", "Analyse", "Paramètres"]:
            item = SidebarItem(name, self.change_page, is_active=(name == self.current_page), mode=self.page.theme_mode)
            self.sidebar_items[name] = item
            sidebar_content.controls.append(item)

        sidebar_content.controls.append(ft.Container(expand=True))
        sidebar_content.controls.append(self.theme_btn)

        self.sidebar = ft.Container(
            content=sidebar_content,
            width=220,
            bgcolor=ft.Colors.TRANSPARENT
        )

        # barre superieure
        self.topbar_title = ft.Text(self.current_page, size=18, weight="bold", color=get_color("text", self.page.theme_mode))
        
        # bouton avec 'content' au lieu de 'text' pour compatibilite
        self.run_btn = ft.OutlinedButton(
            content=ft.Text("▶︎ Lancer"),
            on_click=self.run_backend,
            style=ft.ButtonStyle(
                color=get_color("text", self.page.theme_mode),
                shape=ft.RoundedRectangleBorder(radius=6),
                side=ft.BorderSide(1, get_color("border_color", self.page.theme_mode))
            )
        )

        border_color = get_color("border_color", self.page.theme_mode)

        self.topbar = ft.Container(
            content=ft.Row([self.topbar_title, self.run_btn], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            height=65,
            bgcolor=get_color("container_bg", self.page.theme_mode),
            border_radius=10,
            border=create_border(1, border_color),
            padding=20
        )

        # conteneur principal
        self.main_content = ft.Container(
            bgcolor=get_color("container_bg", self.page.theme_mode),
            border_radius=10,
            border=create_border(1, border_color),
            expand=True
        )

        main_col = ft.Column([self.topbar, self.main_content], expand=True)
        self.layout = ft.Row([self.sidebar, main_col], expand=True)
        
        self.page.add(self.layout)

    def change_page(self, name):
        # changement de page
        if name == self.current_page: return
        self.current_page = name
        
        for item_name, item in self.sidebar_items.items():
            item.set_active(item_name == name, self.page.theme_mode)
            
        self.topbar_title.value = name
        self.run_btn.visible = (name == "Démarrage")
        self.page.update()

    def toggle_theme(self, e):
        # bascule entre mode clair et sombre
        self.page.theme_mode = ft.ThemeMode.DARK if self.page.theme_mode == ft.ThemeMode.LIGHT else ft.ThemeMode.LIGHT
        self.page.bgcolor = get_color("app_bg", self.page.theme_mode)
        
        mode = self.page.theme_mode
        self.theme_btn.bgcolor = get_color("switch_bg", mode)
        self.theme_btn.content.value = "Clair" if mode == ft.ThemeMode.DARK else "Sombre"
        self.theme_btn.content.color = get_color("text", mode)
        
        # changer la couleur du logo
        if mode == ft.ThemeMode.DARK:
            self.logo_img.src = "logo_white.png"
        else:
            self.logo_img.src = "logo_black.png"
        
        for item in self.sidebar_items.values():
            item.set_active(item.is_active, mode)
            
        border_color = get_color("border_color", mode)
            
        self.topbar.bgcolor = get_color("container_bg", mode)
        self.topbar.border = create_border(1, border_color)
        self.topbar_title.color = get_color("text", mode)
        
        self.run_btn.style.color = get_color("text", mode)
        self.run_btn.style.side = ft.BorderSide(1, border_color)

        self.main_content.bgcolor = get_color("container_bg", mode)
        self.main_content.border = create_border(1, border_color)

        self.page.update()

    def run_backend(self, e):
        # demarrage de la logique en arriere-plan
        if self.backend_thread is None or not self.backend_thread.is_alive():
            if BlinkCounter is not None:
                self.backend_instance = BlinkCounter(0)
                # thread en arriere-plan pour ne pas bloquer l'interface
                self.backend_thread = threading.Thread(target=self.backend_instance.process_video, daemon=True)
                self.backend_thread.start()
            else:
                print("Modules backend manquants.")

def main(page: ft.Page):
    App(page)

if __name__ == "__main__":
    # l'argument assets_dir permet a flet de trouver logo.png dans le dossier parent
    ft.run(main, assets_dir="../assets")
