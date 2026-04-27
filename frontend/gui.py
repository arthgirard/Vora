# Migration vers Flet, menu Paramètres et intégration Backend-Frontend codés en partie avec l'aide de Claude.   
import flet as ft
import threading
import asyncio
import sys
import os
import time
import sqlite3
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import io
import base64

# utilisation du backend non-interactif pour flet
matplotlib.use("agg")

# configuration de la police par defaut pour matplotlib avec des fallbacks Linux/Mac
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Segoe UI Variable', 'Segoe UI', 'Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']

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
        # seuil de clignements par minute (10 = faible sensibilité, 15 = haute)
        self.seuil_clignements = 10

        self.build_ui()
        
        # lancement de la boucle d'actualisation des graphiques
        self.page.run_task(self._boucle_analyse)

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

        # page d'analyse, avec un container pour permettre l'expansion du layout grid interne
        self.analyse_content = ft.Container(
            content=self._build_analyse(mode),
            expand=True,
            visible=False,
        )

        # page de paramètres - cachée par défaut
        self.settings_content = ft.Column(
            controls=[self._build_settings(mode)],
            expand=True,
            visible=False,
            scroll=ft.ScrollMode.AUTO,
        )

        # conteneur principal avec pile placeholder / flux vidéo / analyse / paramètres
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
                        content=self.analyse_content,
                        expand=True,
                        padding=ft.Padding.only(top=10, left=10, right=10, bottom=10),
                    ),
                    ft.Container(
                        content=self.settings_content,
                        expand=True,
                        padding=ft.Padding.only(top=10, left=20, right=20, bottom=10),
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

    def _build_analyse(self, mode):
        # ax1 et ax3 = histogrammes (haut), ax2 = nuage de points (bas, large)
        self.fig1, self.ax1 = plt.subplots(figsize=(8, 5), dpi=100)
        self.fig2, self.ax2 = plt.subplots(figsize=(16, 5), dpi=100)
        self.fig3, self.ax3 = plt.subplots(figsize=(8, 5), dpi=100)

        self._style_plots(mode)

        # images pre-generees pour eviter les erreurs au premier affichage
        self.chart1 = ft.Image(src=self._get_chart_base64(self.fig1), expand=True, fit=ft.BoxFit.CONTAIN)
        self.chart2 = ft.Image(src=self._get_chart_base64(self.fig2), expand=True, fit=ft.BoxFit.CONTAIN)
        self.chart3 = ft.Image(src=self._get_chart_base64(self.fig3), expand=True, fit=ft.BoxFit.CONTAIN)

        border_color = get_color("border_color", mode)
        bg = get_color("container_bg", mode)

        # cartes qui encadrent chaque graphique, cohérentes avec le reste de l'interface
        def make_card(chart):
            return ft.Container(
                content=chart,
                expand=True,
                bgcolor=bg,
                border=create_border(1, border_color),
                border_radius=10,
                padding=8,
                alignment=ft.Alignment(0, 0),
            )

        self.card1 = make_card(self.chart1)
        self.card2 = make_card(self.chart2)
        self.card3 = make_card(self.chart3)

        # haut: deux histogrammes cote a cote — bas: nuage de points pleine largeur
        row_top = ft.Row([self.card1, self.card3], expand=True, spacing=10)
        row_bottom = ft.Row([self.card2], expand=True)

        return ft.Column([row_top, row_bottom], expand=True, spacing=10)

    def _style_plots(self, mode):
        # couleurs issues du theme actif
        text_color = get_color("text", mode)
        subtle_color = get_color("text_subtle", mode)
        border_color = get_color("border_color", mode)
        bg = get_color("container_bg", mode)
        is_dark = (mode == ft.ThemeMode.DARK)

        # grilles et labels d'axes : noir en mode clair, blanc en mode sombre
        axis_color = "#FFFFFF" if is_dark else "#000000"
        grid_color = "#3D3D3D" if is_dark else "#E6E6E6"

        # ax1 et ax3 = histogrammes, ax2 = nuage de points (bas, large)
        configs = [
            (self.ax1, "Fréquence des clignements", "Clignements / minute", "Fréquence"),
            (self.ax2, "Fiabilité de la détection", "Minute",               "Fiabilité"),
            (self.ax3, "Alertes de fatigue",         "Niveau de fatigue",   "Fréquence"),
        ]

        # les figures font ~2.35x la taille affichee — toutes les tailles sont scalees en consequence
        for ax, titre, xlabel, ylabel in configs:
            ax.set_title(titre, fontsize=22, fontweight='semibold', color=text_color, pad=28)
            ax.set_xlabel(xlabel, fontsize=20, color=axis_color, labelpad=18)
            ax.set_ylabel(ylabel, fontsize=20, color=axis_color, labelpad=18)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(border_color)
            ax.spines['left'].set_linewidth(2.0)
            ax.spines['bottom'].set_color(border_color)
            ax.spines['bottom'].set_linewidth(2.0)

            # rotation et alignement de l'axe x pour eviter le chevauchement
            ax.tick_params(colors=axis_color, labelsize=15, length=8, width=2.0)
            plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
            
            ax.set_facecolor(bg)

            ax.grid(True, color=grid_color, linewidth=1.8, alpha=1.0)
            ax.set_axisbelow(True)

        # axe X de la fiabilite commence a 1 (les minutes ne peuvent pas descendre en dessous)
        self.ax2.set_ylim(-0.1, 1.1)
        self.ax2.set_xlim(left=1)

        for fig in [self.fig1, self.fig2, self.fig3]:
            fig.patch.set_facecolor(bg)
            fig.tight_layout(pad=2.5)

    def _get_chart_base64(self, fig):
        # sauvegarde propre d'une figure au format data uri
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def _refresh_chart_images(self, update=True):
        # mise a jour du parametre src avec la version encodee la plus recente
        for fig, chart in zip([self.fig1, self.fig2, self.fig3], [self.chart1, self.chart2, self.chart3]):
            chart.src = self._get_chart_base64(fig)
            if update and chart.page:
                chart.update()

    def _update_graphs(self):
        # recherche robuste pour trouver la base de donnees, peu importe d'ou le script est lance
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_paths = [
            os.path.join(base_dir, 'backend', 'data_blinks.db'),
            os.path.join(base_dir, 'frontend', 'data_blinks.db'),
            os.path.join(base_dir, 'data_blinks.db'),
            os.path.abspath('data_blinks.db')
        ]
        
        db_path = next((p for p in db_paths if os.path.exists(p)), None)

        if not db_path:
            return

        try:
            # timeout pour eviter les erreurs si le backend ecrit en meme temps
            conn = sqlite3.connect(db_path, timeout=5)
            cursor = conn.cursor()
            
            # selection de la table la plus recente
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name DESC LIMIT 1;")
            tables = cursor.fetchall()
            
            if not tables:
                conn.close()
                return
            
            table_name = tables[0][0]
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            conn.close()

            if df.empty:
                return

            x = df.iloc[:, 0]
            accent_color = get_color("accent", self.page.theme_mode)

            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()

            # securite: on s'assure que les colonnes existent avant de les dessiner
            if len(df.columns) > 1:
                data1 = df.iloc[:, 1].dropna()
                bins1 = max(1, min(20, len(data1)))
                self.ax1.hist(data1, bins=bins1, color=accent_color, edgecolor='none', alpha=1.0, rwidth=0.75, zorder=3)
            if len(df.columns) > 2:
                green = "#2ECC71"
                self.ax2.plot(x, df.iloc[:, 2], color=green, linewidth=4.5, zorder=3)
                self.ax2.scatter(x, df.iloc[:, 2], color=green, s=180, zorder=4, edgecolors='none')
                self.ax2.fill_between(x, df.iloc[:, 2], alpha=0.10, color=green, zorder=2)
                if len(x) > 0:
                    self.ax2.set_xlim(left=min(float(x.min()), 1))
            if len(df.columns) > 3:
                red = "#E74C3C"
                data3 = df.iloc[:, 3].dropna()
                bins3 = max(1, min(20, len(data3)))
                self.ax3.hist(data3, bins=bins3, color=red, edgecolor='none', alpha=1.0, rwidth=0.75, zorder=3)

            self._style_plots(self.page.theme_mode)
            self._refresh_chart_images()

        except Exception as e:
            print(f"Erreur d'actualisation des graphiques: {e}")

    async def _boucle_analyse(self):
        # boucle asynchrone pour rafraichir les graphiques
        while True:
            if self.current_page == "Analyse":
                self._update_graphs()
            await asyncio.sleep(2)

    def _build_settings(self, mode):
        # construit le contenu de la page paramètres et stocke les refs nécessaires au thème
        border_color = get_color("border_color", mode)

        # switch du thème - remplace l'ancien bouton pillule
        self.theme_switch = ft.Switch(value=(mode == ft.ThemeMode.DARK), active_color=get_color("accent", mode))
        self.theme_switch.on_change = self.toggle_theme

        # textes des lignes - stockés pour mise à jour du thème
        self.settings_label_apparence = ft.Text("Apparence", size=11, weight=ft.FontWeight.W_600, color=get_color("text_subtle", mode))
        self.settings_label_mode_sombre = ft.Text("Mode sombre", size=13, color=get_color("text", mode), expand=True)

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

        # section détection : slider de sensibilité des alertes de clignement
        self.settings_label_detection = ft.Text("Détection", size=11, weight=ft.FontWeight.W_600, color=get_color("text_subtle", mode))
        self.settings_label_sensibilite = ft.Text("Sensibilité des alertes de clignement", size=13, color=get_color("text", mode))
        self.settings_label_seuil_valeur = ft.Text(
            f"Seuil : {self.seuil_clignements} clignements / minute",
            size=12,
            color=get_color("text_subtle", mode)
        )

        self.seuil_slider = ft.Slider(
            min=10, max=15, divisions=5, value=self.seuil_clignements,
            active_color=get_color("accent", mode),
            inactive_color=get_color("border_color", mode),
            expand=True,
            on_change=self._on_seuil_change,
        )

        self.settings_section_detection = ft.Container(
            content=ft.Column([
                self.settings_label_detection,
                ft.Container(height=8),
                ft.Row([self.settings_label_sensibilite], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                self.settings_label_seuil_valeur,
                ft.Container(height=4),
                ft.Row([
                    ft.Text("Faible", size=11, color=get_color("text_subtle", mode)),
                    self.seuil_slider,
                    ft.Text("Haute",  size=11, color=get_color("text_subtle", mode)),
                ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ], spacing=0),
            bgcolor=get_color("container_bg", mode),
            border=create_border(1, border_color),
            border_radius=8,
            padding=16,
        )

        return ft.Column(
            [self.settings_section_apparence, ft.Container(height=12), self.settings_section_detection],
            spacing=0,
        )

    def _on_seuil_change(self, e):
        # mise a jour du seuil et de l'etiquette affichee
        self.seuil_clignements = int(e.control.value)
        self.settings_label_seuil_valeur.value = f"Seuil : {self.seuil_clignements} clignements / minute"
        self.settings_label_seuil_valeur.update()
        if self.backend_instance:
            self.backend_instance.seuil_clignements = self.seuil_clignements

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
        self.analyse_content.visible = False
        self.placeholder.visible = False
        self.video_feed.visible = True
        self.main_content.update()

    def _afficher_placeholder(self):
        # bascule de l'affichage vers le placeholder
        self.settings_content.visible = False
        self.analyse_content.visible = False
        self.video_feed.visible = False
        self.placeholder.visible = True
        self.main_content.update()

    def _afficher_parametres(self):
        # bascule de l'affichage vers la page de paramètres
        self.video_feed.visible = False
        self.analyse_content.visible = False
        self.placeholder.visible = False
        self.settings_content.visible = True
        self.main_content.update()

    def _afficher_analyse(self):
        # bascule de l'affichage vers la page d'analyse
        self.video_feed.visible = False
        self.placeholder.visible = False
        self.settings_content.visible = False
        self.analyse_content.visible = True
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
        elif name == "Analyse":
            self.affichage_pause = True
            self._afficher_analyse()
            self._update_graphs() # force la mise a jour
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
        self.settings_label_mode_sombre.color = get_color("text", mode)
        self.settings_section_apparence.bgcolor = get_color("container_bg", mode)
        self.settings_section_apparence.border = create_border(1, border_color)

        self.settings_label_detection.color = subtle
        self.settings_label_sensibilite.color = get_color("text", mode)
        self.settings_label_seuil_valeur.color = subtle
        self.seuil_slider.active_color = get_color("accent", mode)
        self.seuil_slider.inactive_color = border_color
        self.settings_section_detection.bgcolor = get_color("container_bg", mode)
        self.settings_section_detection.border = create_border(1, border_color)

        # mise a jour du theme des graphiques et des cartes qui les entourent
        if hasattr(self, 'fig1'):
            for card in [self.card1, self.card2, self.card3]:
                card.bgcolor = get_color("container_bg", mode)
                card.border = create_border(1, border_color)
            self._style_plots(mode)
            self._refresh_chart_images()

        self.page.update()

    def run_backend(self, e):
        # demarrage ou arret avec changement du bouton
        if self.backend_instance and getattr(self.backend_instance, 'running', False):
            self.backend_instance.stop()
            self.run_btn.content.value = "▶︎ Lancer"
            self.run_btn.style.color = get_color("text", self.page.theme_mode)
            self._afficher_placeholder()
        else:
            if BlinkCounter is not None:
                self.run_btn.content.value = "■ Arrêter"
                self.run_btn.style.color = ft.Colors.RED_400
                self._afficher_flux()
                # instanciation du backend
                self.backend_instance = BlinkCounter()
                
                # Transfert immédiat du seuil sélectionné dans Flet
                self.backend_instance.seuil_clignements = self.seuil_clignements
                
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
