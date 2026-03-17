class PostAnalysisRouter:
    """
    A router to control all database operations on models in the
    post_analysis application.
    """
    route_app_labels = {'post_analysis'}
    recommendation_models = {'metricoptionrecommendation', 'metricgradechecksetting', 'metricpercentbasesetting'}

    def db_for_read(self, model, **hints):
        """
        Attempts to read post_analysis models go to default db.
        """
        if model._meta.app_label in self.route_app_labels:
            if model._meta.model_name in self.recommendation_models:
                return 'post_analysis_recommendations'
            return 'default'
        return None

    def db_for_write(self, model, **hints):
        """
        Attempts to write post_analysis models go to default db.
        """
        if model._meta.app_label in self.route_app_labels:
            if model._meta.model_name in self.recommendation_models:
                return 'post_analysis_recommendations'
            return 'default'
        return None

    def allow_relation(self, obj1, obj2, **hints):
        """
        Allow relations if a model in the post_analysis app is involved.
        """
        if (
            obj1._meta.app_label in self.route_app_labels or
            obj2._meta.app_label in self.route_app_labels
        ):
            return True
        return None

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        """
        Make sure the post_analysis app only appears in the 'default' database.
        """
        if app_label in self.route_app_labels:
            if model_name in self.recommendation_models:
                return db == 'post_analysis_recommendations'
            return db == 'default'
        return None
