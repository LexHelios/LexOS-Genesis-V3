# ... [earlier code omitted for brevity] ...

class EnhancedWebInterface(WebInterface):
    """Enhanced web interface with authentication and file uploads"""

    def __init__(self, lexos, config: ConfigManager):
        self.config = config
        super().__init__(lexos)
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(exist_ok=True)
        self._setup_middleware()

    def _setup_routes(self):
        """Setup enhanced routes"""
        super()._setup_routes()

        # Additional routes
        self.app.router.add_post('/api/upload', self.upload_file)
        self.app.router.add_get('/api/agents', self.get_agents)
        self.app.router.add_post('/api/config', self.update_config)
        self.app.router.add_get('/api/performance', self.get_performance)
        self.app.router.add_get('/api/errors', self.get_errors)
        self.app.router.add_post('/api/backup', self.create_backup)

        # Existing static: serve from / for legacy dashboards
        self.app.router.add_static('/', path='static', name='static')

        # --- SPA Static & Catch-All ---
        # Serve static files for SPA from /static/
        spa_dist_path = os.path.abspath('frontend/dist')
        if not os.path.isdir(spa_dist_path):
            # Production hardening: fail gracefully if SPA build is missing
            logger.warning(f"SPA build directory not found at {spa_dist_path}. Static SPA routes will not be served.")
        else:
            self.app.router.add_static('/static/', path=spa_dist_path, name='spa-static')

            async def index_handler(request):
                index_path = os.path.join(spa_dist_path, 'index.html')
                if os.path.exists(index_path):
                    return web.FileResponse(index_path)
                else:
                    # Fallback: minimal HTML if SPA build missing
                    return web.Response(text="<html><body><h3>LexOS SPA not built. Please run frontend build.</h3></body></html>", content_type='text/html')

            # Catch-all for SPA client-side routes
            self.app.router.add_get('/{tail:.*}', index_handler)

    # ... [rest of EnhancedWebInterface unchanged] ...
