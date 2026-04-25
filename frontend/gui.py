# Migration vers Flet, menu Paramètres et intégration Backend-Frontend codés en partie avec l'aide de Claude.   
import flet as ft
import threading
import asyncio
import glob
import platform
import subprocess
import sys
import os
import time

# ajout du chemin du backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
try:
    from blink_counter import BlinkCounter
except ImportError as e:
    print(f"Impossible d'importer BlinkCounter : {e}")
    BlinkCounter = None

COLORS = {
    "app_bg": {ft.ThemeMode.LIGHT: "#F3F3F3", ft.ThemeMode.DARK: "#202020"},
    "container_bg": {ft.ThemeMode.LIGHT: "#FFFFFF", ft.ThemeMode.DARK: "#1C1C1C"},
    "hover_bg": {ft.ThemeMode.LIGHT: "#E5E5E5", ft.ThemeMode.DARK: "#2D2D2D"},
    "border_color": {ft.ThemeMode.LIGHT: "#E0E0E0", ft.ThemeMode.DARK: "#2D2D2D"},
    "active_bg": {ft.ThemeMode.LIGHT: "#EBEBEB", ft.ThemeMode.DARK: "#323232"},
    "accent": "#0078D4",
    "text": {ft.ThemeMode.LIGHT: "#000000", ft.ThemeMode.DARK: "#FFFFFF"},
    "text_subtle": {ft.ThemeMode.LIGHT: "#999999", ft.ThemeMode.DARK: "#555555"},
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

def _noms_linux():
    # lit les noms depuis sysfs - rapide, sans ouvrir aucun périphérique
    noms = {}
    for path in glob.glob('/sys/class/video4linux/video*/name'):
        try:
            idx = int(path.split('video4linux/video')[1].split('/')[0])
            with open(path) as f:
                noms[idx] = f.read().strip()
        except (OSError, ValueError, IndexError):
            pass

    # fallback v4l2-ctl pour les systèmes sans sysfs exposé
    if not noms:
        try:
            out = subprocess.check_output(
                ['v4l2-ctl', '--list-devices'],
                stderr=subprocess.DEVNULL, timeout=2
            ).decode(errors='ignore')
            nom_courant = None
            for line in out.splitlines():
                line = line.strip()
                if line and not line.startswith('/dev/'):
                    nom_courant = line.rstrip(':')
                elif line.startswith('/dev/video'):
                    try:
                        idx = int(line.replace('/dev/video', ''))
                        if nom_courant:
                            noms[idx] = nom_courant
                    except ValueError:
                        pass
        except Exception:
            pass
    return noms

def _noms_windows():
    # récupère les noms de caméras via PowerShell sans dépendance externe
    noms = {}
    try:
        out = subprocess.check_output(
            ['powershell', '-Command',
             'Get-PnpDevice -Class Camera | Select-Object -ExpandProperty FriendlyName'],
            stderr=subprocess.DEVNULL, timeout=3
        ).decode('utf-8', errors='ignore')
        for i, line in enumerate(l.strip() for l in out.splitlines() if l.strip()):
            noms[i] = line
    except Exception:
        pass
    return noms

def _noms_macos():
    # récupère les noms via system_profiler - présent sur tous les Mac
    noms = {}
    try:
        out = subprocess.check_output(
            ['system_profiler', 'SPCameraDataType'],
            stderr=subprocess.DEVNULL, timeout=3
        ).decode(errors='ignore')
        idx = 0
        for line in out.splitlines():
            stripped = line.strip().rstrip(':')
            # les noms de caméra sont des lignes non indentées suivies de ':'
            if stripped and line.startswith('    ') and line.endswith(':'):
                noms[idx] = stripped
                idx += 1
    except Exception:
        pass
    return noms

def detecter_cameras():
    # retourne une liste de tuples (index, nom) adaptée à la plateforme courante sans utiliser OpenCV
    systeme = platform.system()
    noms = {}

    if systeme == 'Linux':
        noms = _noms_linux()
    elif systeme == 'Windows':
        noms = _noms_windows()
    elif systeme == 'Darwin':
        noms = _noms_macos()

    resultats = []
    if noms:
        for idx, nom in noms.items():
            resultats.append((idx, nom))
    else:
        # Fallback ultra sécurisé : on liste juste des index bruts si l'OS ne répond pas
        # pour éviter d'utiliser cv.VideoCapture() et de bloquer le driver.
        for idx in range(3):
            resultats.append((idx, f"Caméra {idx}"))

    return resultats if resultats else [(0, 'Caméra par défaut')]

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
        # pause de l'affichage quand l'utilisateur change d'onglet
        self.affichage_pause = False
        # index de la caméra sélectionnée dans les paramètres
        self.camera_index = 0

        self.build_ui()

    def build_ui(self):
        mode = self.page.theme_mode

        # création du logo avec une référence pour changer la couleur plus tard
        self.logo_img = ft.Image(
            src="logo_black.png",
            height=40,
            fit=ft.BoxFit.CONTAIN,
        )

        # barre laterale sans le toggle de thème (déplacé dans les paramètres)
        sidebar_content = ft.Column(
            controls=[
                ft.Container(content=self.logo_img, padding=10)
            ],
            expand=True
        )

        for name in ["Démarrage", "Analyse", "Paramètres"]:
            item = SidebarItem(name, self.change_page, is_active=(name == self.current_page), mode=mode)
            self.sidebar_items[name] = item
            sidebar_content.controls.append(item)

        sidebar_content.controls.append(ft.Container(expand=True))

        self.sidebar = ft.Container(
            content=sidebar_content,
            width=220,
            bgcolor=ft.Colors.TRANSPARENT
        )

        # barre superieure
        self.topbar_title = ft.Text(self.current_page, size=18, weight="bold", color=get_color("text", mode))

        self.run_btn = ft.OutlinedButton(
            content=ft.Text("▶︎ Lancer"),
            on_click=self.run_backend,
            style=ft.ButtonStyle(
                color=get_color("text", mode),
                shape=ft.RoundedRectangleBorder(radius=6),
                side=ft.BorderSide(1, get_color("border_color", mode))
            )
        )

        border_color = get_color("border_color", mode)

        self.topbar = ft.Container(
            content=ft.Row([self.topbar_title, self.run_btn], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            height=65,
            bgcolor=get_color("container_bg", mode),
            border_radius=10,
            border=create_border(1, border_color),
            padding=20
        )

        # placeholder affiché quand la caméra est éteinte
        self.placeholder_icon = ft.Text("◉", size=36, color=get_color("text_subtle", mode))
        self.placeholder_titre = ft.Text(
            "Vora n'est pas en cours",
            size=15,
            weight=ft.FontWeight.W_600,
            color=get_color("text_subtle", mode)
        )
        self.placeholder_sous_titre = ft.Text(
            "Appuyez sur Lancer pour démarrer la détection",
            size=12,
            color=get_color("text_subtle", mode)
        )
        self.placeholder = ft.Column(
            [self.placeholder_icon, self.placeholder_titre, self.placeholder_sous_titre],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=8,
            expand=True,
        )

        # flux vidéo - caché par défaut, gapless_playback évite le flash entre les frames
        self.video_feed = ft.Image(
            src="logo_black.png",
            visible=False,
            border_radius=8,
            fit=ft.BoxFit.CONTAIN,
            gapless_playback=True,
            expand=True,
        )

        # page de paramètres - cachée par défaut
        self.settings_content = ft.Column(
            controls=[self._build_settings(mode)],
            expand=True,
            visible=False,
            scroll=ft.ScrollMode.AUTO,
        )

        # conteneur principal avec pile placeholder / flux vidéo / paramètres
        self.main_content = ft.Container(
            content=ft.Stack(
                [
                    ft.Container(
                        content=self.placeholder,
                        expand=True,
                        alignment=ft.Alignment(0, 0),
                    ),
                    ft.Container(
                        content=self.video_feed,
                        expand=True,
                        border_radius=8,
                        clip_behavior=ft.ClipBehavior.HARD_EDGE,
                    ),
                    ft.Container(
                        content=self.settings_content,
                        expand=True,
                        padding=ft.padding.only(top=10, left=20, right=20, bottom=10),
                    ),
                ],
                expand=True,
            ),
            bgcolor=get_color("container_bg", mode),
            border_radius=10,
            border=create_border(1, border_color),
            expand=True,
            padding=10,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
        )

        main_col = ft.Column([self.topbar, self.main_content], expand=True)
        self.layout = ft.Row([self.sidebar, main_col], expand=True)

        self.page.add(self.layout)

    def _build_settings(self, mode):
        # construit le contenu de la page paramètres et stocke les refs nécessaires au thème
        border_color = get_color("border_color", mode)

        # switch du thème - remplace l'ancien bouton pillule
        self.theme_switch = ft.Switch(value=(mode == ft.ThemeMode.DARK), active_color=get_color("accent", mode))
        self.theme_switch.on_change = self.toggle_theme

        # menu déroulant de sélection de la caméra - les tuples (idx, nom) viennent de detecter_cameras
        cameras = detecter_cameras()
        options = []
        for idx, nom in cameras:
            label = f"{nom}  (/dev/video{idx})" if platform.system() == 'Linux' else nom
            options.append(ft.dropdown.Option(str(idx), label))
            
        self.camera_dropdown = ft.Dropdown(
            options=options,
            value=str(self.camera_index),
            width=220,
            border_color=border_color,
            border_radius=6,
            text_size=13,
            color=get_color("text", mode),
        )
        self.camera_dropdown.on_change = self.changer_camera

        # textes des lignes - stockés pour mise à jour du thème
        self.settings_label_apparence = ft.Text("Apparence", size=11, weight=ft.FontWeight.W_600, color=get_color("text_subtle", mode))
        self.settings_label_camera_section = ft.Text("Caméra", size=11, weight=ft.FontWeight.W_600, color=get_color("text_subtle", mode))
        self.settings_label_mode_sombre = ft.Text("Mode sombre", size=13, color=get_color("text", mode), expand=True)
        self.settings_label_periph = ft.Text("Périphérique actif", size=13, color=get_color("text", mode), expand=True)

        # cartes des sections - stockées pour mise à jour du thème
        self.settings_section_apparence = ft.Container(
            content=ft.Column([
                self.settings_label_apparence,
                ft.Container(height=8),
                ft.Row([self.settings_label_mode_sombre, self.theme_switch], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ], spacing=0),
            bgcolor=get_color("container_bg", mode),
            border=create_border(1, border_color),
            border_radius=8,
            padding=16,
        )
        self.settings_note_camera = ft.Text(
            "Changer de caméra redémarre automatiquement la détection.",
            size=11,
            color=get_color("text_subtle", mode),
            italic=True,
        )
        self.settings_section_camera = ft.Container(
            content=ft.Column([
                self.settings_label_camera_section,
                ft.Container(height=8),
                ft.Row([self.settings_label_periph, self.camera_dropdown], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Container(height=6),
                self.settings_note_camera,
            ], spacing=0),
            bgcolor=get_color("container_bg", mode),
            border=create_border(1, border_color),
            border_radius=8,
            padding=16,
        )

        return ft.Column(
            [self.settings_section_apparence, ft.Container(height=12), self.settings_section_camera],
            spacing=0,
        )

    def changer_camera(self, e):
        new_index = int(e.control.value)
        self.camera_index = new_index
        # Lancement via un simple Thread de Python pour être immunisé contre les exceptions silencieuses
        threading.Thread(target=self._worker_change_cam, args=(new_index,), daemon=True).start()

    def _worker_change_cam(self, new_index):
        # si le backend tourne déjà, le redémarrer avec la nouvelle caméra
        if self.backend_instance and getattr(self.backend_instance, 'running', False):
            
            # Mise à jour de l'UI manuelle depuis le thread (Flet l'autorise)
            self.settings_note_camera.value = f"🔄 Redémarrage sur la caméra {new_index} en cours..."
            self.settings_note_camera.color = get_color("accent", self.page.theme_mode)
            try:
                self.settings_note_camera.update()
            except Exception:
                pass
            
            # Couper le backend
            self.backend_instance.stop()
            
            if self.backend_thread and self.backend_thread.is_alive():
                self.backend_thread.join(timeout=3.0)
            
            # Laisser le temps à l'OS de réinitialiser le port USB
            time.sleep(1.0)
            
            self.backend_instance = None
            
            if BlinkCounter is not None:
                self.backend_instance = BlinkCounter(self.camera_index)
                self.backend_thread = threading.Thread(target=self.backend_instance.process_video, daemon=True)
                self.backend_thread.start()
                
                # Relancer la tâche de mise à jour d'image
                self.page.run_task(self._boucle_affichage, self.backend_instance)
                
                # Feedback de réussite
                self.settings_note_camera.value = "✅ Caméra modifiée avec succès !"
                self.settings_note_camera.color = ft.Colors.GREEN
        else:
            # Succès si la caméra est à l'arrêt
            self.settings_note_camera.value = f"✅ Caméra {new_index} sélectionnée. Prête pour le lancement."
            self.settings_note_camera.color = ft.Colors.GREEN

        # Mise à jour visuelle du succès
        try:
            self.settings_note_camera.update()
        except Exception:
            pass
            
        # Restauration du texte d'origine après 2 secondes
        time.sleep(2.0)
        self.settings_note_camera.value = "Changer de caméra redémarre automatiquement la détection."
        self.settings_note_camera.color = get_color("text_subtle", self.page.theme_mode)
        try:
            self.settings_note_camera.update()
        except Exception:
            pass

    async def _boucle_affichage(self, instance_cible):
        # La boucle vérifie désormais SON instance, et s'arrêtera proprement
        while instance_cible and getattr(instance_cible, 'running', False):
            if not self.affichage_pause:
                with instance_cible._frame_lock:
                    frame = instance_cible.latest_frame
                if frame:
                    self.video_feed.src = f"data:image/jpeg;base64,{frame}"
                    self.video_feed.update()
            await asyncio.sleep(0.05)

    def _afficher_flux(self):
        # bascule de l'affichage vers le flux vidéo
        self.settings_content.visible = False
        self.placeholder.visible = False
        self.video_feed.visible = True
        self.main_content.update()

    def _afficher_placeholder(self):
        # bascule de l'affichage vers le placeholder
        self.settings_content.visible = False
        self.video_feed.visible = False
        self.placeholder.visible = True
        self.main_content.update()

    def _afficher_parametres(self):
        # bascule de l'affichage vers la page de paramètres
        self.video_feed.visible = False
        self.placeholder.visible = False
        self.settings_content.visible = True
        self.main_content.update()

    def change_page(self, name):
        # changement de page
        if name == self.current_page: return
        self.current_page = name

        for item_name, item in self.sidebar_items.items():
            item.set_active(item_name == name, self.page.theme_mode)

        self.topbar_title.value = name
        self.run_btn.visible = (name == "Démarrage")

        if name == "Démarrage":
            # reprendre le flux s'il était actif
            self.affichage_pause = False
            if self.backend_instance and getattr(self.backend_instance, 'running', False):
                self._afficher_flux()
            else:
                self._afficher_placeholder()
        elif name == "Paramètres":
            # suspendre le flux et afficher les paramètres
            self.affichage_pause = True
            self._afficher_parametres()
        else:
            # suspendre le flux pour les autres onglets
            self.affichage_pause = True
            self._afficher_placeholder()

        self.page.update()

    def toggle_theme(self, e):
        # bascule entre mode clair et sombre - appelé par le ft.Switch dans les paramètres
        self.page.theme_mode = ft.ThemeMode.DARK if self.page.theme_mode == ft.ThemeMode.LIGHT else ft.ThemeMode.LIGHT
        self.page.bgcolor = get_color("app_bg", self.page.theme_mode)

        mode = self.page.theme_mode

        # synchroniser la valeur du switch si la bascule vient d'ailleurs
        self.theme_switch.value = (mode == ft.ThemeMode.DARK)

        # changer la couleur du logo
        self.logo_img.src = "logo_white.png" if mode == ft.ThemeMode.DARK else "logo_black.png"

        for item in self.sidebar_items.values():
            item.set_active(item.is_active, mode)

        border_color = get_color("border_color", mode)
        subtle = get_color("text_subtle", mode)

        self.topbar.bgcolor = get_color("container_bg", mode)
        self.topbar.border = create_border(1, border_color)
        self.topbar_title.color = get_color("text", mode)

        self.run_btn.style.color = get_color("text", mode)
        self.run_btn.style.side = ft.BorderSide(1, border_color)

        self.main_content.bgcolor = get_color("container_bg", mode)
        self.main_content.border = create_border(1, border_color)

        # mettre à jour les couleurs du placeholder
        self.placeholder_icon.color = subtle
        self.placeholder_titre.color = subtle
        self.placeholder_sous_titre.color = subtle

        # mettre à jour les couleurs des sections de paramètres
        self.settings_label_apparence.color = subtle
        self.settings_label_camera_section.color = subtle
        self.settings_label_mode_sombre.color = get_color("text", mode)
        self.settings_label_periph.color = get_color("text", mode)
        self.settings_section_apparence.bgcolor = get_color("container_bg", mode)
        self.settings_section_apparence.border = create_border(1, border_color)
        self.settings_section_camera.bgcolor = get_color("container_bg", mode)
        self.settings_section_camera.border = create_border(1, border_color)
        self.camera_dropdown.color = get_color("text", mode)
        self.camera_dropdown.border_color = border_color
        self.settings_note_camera.color = get_color("text_subtle", mode)

        self.page.update()

    def run_backend(self, e):
        # demarrage ou arret avec changement du bouton
        if self.backend_instance and getattr(self.backend_instance, 'running', False):
            self.backend_instance.stop()
            self.run_btn.content.value = "▶︎ Lancer"
            self.run_btn.style.color = get_color("text", self.page.theme_mode)
            # rien à réactiver - le dropdown reste toujours actif
            self._afficher_placeholder()
        else:
            if BlinkCounter is not None:
                self.run_btn.content.value = "■ Arrêter"
                self.run_btn.style.color = ft.Colors.RED_400
                # le dropdown reste actif - changer la caméra redémarre le backend automatiquement
                self._afficher_flux()
                # instanciation du backend avec la caméra sélectionnée dans les paramètres
                self.backend_instance = BlinkCounter(self.camera_index)
                # thread en arriere-plan pour ne pas bloquer l'interface
                self.backend_thread = threading.Thread(target=self.backend_instance.process_video, daemon=True)
                self.backend_thread.start()
                # boucle async lancée dans l'event loop de Flet pour afficher les frames
                self.page.run_task(self._boucle_affichage, self.backend_instance)
            else:
                print("Modules backend manquants.")
        self.page.update()

def main(page: ft.Page):
    App(page)

if __name__ == "__main__":
    # l'argument assets_dir permet a flet de trouver logo.png dans le dossier parent
    ft.run(main, assets_dir="../assets")
