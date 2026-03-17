"""
PlankInspection – Windows Service wrapper (pywin32).

Kör hela FastAPI-appen (main.py) som en äkta Windows-tjänst.
Installera:   service.exe install
Starta:       service.exe start       (eller net start PlankInspection)
Stoppa:       service.exe stop
Avinstallera: service.exe remove
"""

import os
import sys
import time
import threading
import logging

import win32serviceutil
import win32service
import win32event
import servicemanager


# ============================================================
# 1) Fixa working directory
#    En Windows-tjänst startar i C:\Windows\System32 —
#    vi måste peka på installationsmappen så att modeller,
#    templates, static-filer och config hittas.
# ============================================================

def get_install_dir():
    """Returnera mappen där service.exe (eller service.py) ligger."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 2) Loggning till fil (tjänster har ingen konsol)
# ============================================================

def setup_logging(install_dir):
    """Konfigurera loggning till fil."""
    # Använd DATA_DIR om tillgängligt, annars install_dir
    from config import DATA_DIR
    log_dir = os.path.join(DATA_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "service.log")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("PlankService")


# ============================================================
# 3) Tjänsteklass
# ============================================================

class PlankInspectionService(win32serviceutil.ServiceFramework):

    _svc_name_ = "PlankInspection"
    _svc_display_name_ = "Plank Inspection Service"
    _svc_description_ = (
        "Kör plankinspektion med AI-baserad bildanalys. "
        "Webbpanel nås på http://<ip>:<port> (se settings.ini)"
    )

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)
        self.server_thread = None
        self.server = None

    # ----- Tjänsten startar -----
    def SvcDoRun(self):
        """Kallas av Windows när tjänsten ska starta."""
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, ""),
        )

        # Sätt working directory
        install_dir = get_install_dir()
        os.chdir(install_dir)

        logger = setup_logging(install_dir)
        logger.info(f"Tjänsten startar. Installationsdir: {install_dir}")

        try:
            self._run_server(logger)
        except Exception as e:
            logger.exception(f"Oväntat fel: {e}")

    def _run_server(self, logger):
        """Starta uvicorn i en egen tråd och vänta på stopp-signal."""
        import uvicorn
        from config import config

        # Windows-tjänster har ingen konsol → sys.stdout/stderr är None.
        if sys.stdout is None:
            sys.stdout = open(os.devnull, "w")
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w")

        uvi_config = uvicorn.Config(
            app="main:app",
            host=config.HOST,
            port=config.PORT,
            log_level="info",
            log_config=None,
            access_log=False,
        )
        self.server = uvicorn.Server(uvi_config)

        self.server_thread = threading.Thread(
            target=self.server.run,
            daemon=True,
        )
        self.server_thread.start()
        logger.info(f"Uvicorn startad på {config.HOST}:{config.PORT}")

        # Vänta tills Windows säger åt oss att stanna
        win32event.WaitForSingleObject(self.stop_event, win32event.INFINITE)

        logger.info("Stoppbeordrad — stänger uvicorn...")
        self.server.should_exit = True
        self.server_thread.join(timeout=10)
        logger.info("Tjänsten stoppad.")

    # ----- Tjänsten stoppas -----
    def SvcStop(self):
        """Kallas av Windows när tjänsten ska stoppas."""
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.stop_event)


# ============================================================
# 4) Entry point
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Inga argument → startad av Windows SCM → kör som tjänst
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(PlankInspectionService)
        servicemanager.StartServiceCtrlDispatcher()

    elif sys.argv[1].lower() == "debug":
        # "service.exe debug" → kör i terminal utan tjänst
        print("Running in debug mode (not as Windows Service)...")
        install_dir = get_install_dir()
        os.chdir(install_dir)
        logger = setup_logging(install_dir)
        logger.info("Debug mode started. Dir: %s", install_dir)

        import uvicorn
        from config import config

        print(f"Starting server on {config.HOST}:{config.PORT}")
        logger.info("Uvicorn starting on %s:%s", config.HOST, config.PORT)
        uvicorn.run("main:app",
                    host=config.HOST,
                    port=config.PORT,
                    log_level="info",
                    reload=False)
    else:
        # Argument (install/start/stop/remove) → Windows Service
        win32serviceutil.HandleCommandLine(PlankInspectionService)