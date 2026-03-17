class HostedBackendRouter:
    """
    A router to control database operations on models in the
    hosted_backend_config application.
    """
    route_app_labels = {'hosted_backend_config'}

    def db_for_read(self, model, **hints):
        """
        Attempts to read hosted_backend_config models go to hosted_backend_db.
        """
        if model._meta.app_label in self.route_app_labels:
            return 'hosted_backend_db'
        return None

    def db_for_write(self, model, **hints):
        """
        Attempts to write hosted_backend_config models go to hosted_backend_db.
        """
        if model._meta.app_label in self.route_app_labels:
            return 'hosted_backend_db'
        return None

    def allow_relation(self, obj1, obj2, **hints):
        """
        Allow relations if a model in the hosted_backend_config app is involved.
        """
        if (
            obj1._meta.app_label in self.route_app_labels or
            obj2._meta.app_label in self.route_app_labels
        ):
            return True
        return None

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        """
        Make sure the hosted_backend_config app only appears in the 'hosted_backend_db'
        database.
        """
        if app_label in self.route_app_labels:
            return db == 'hosted_backend_db'
        return None
