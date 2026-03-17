class PersonalRankingRouter:
    """
    Router to control all database operations on models in
    the personal_ranking_list_app application
    """
    app_label = 'personal_ranking_list_app'

    def db_for_read(self, model, **hints):
        if model._meta.app_label == self.app_label:
            return 'personal_ranking'
        return None

    def db_for_write(self, model, **hints):
        if model._meta.app_label == self.app_label:
            return 'personal_ranking'
        return None

    def allow_relation(self, obj1, obj2, **hints):
        if (obj1._meta.app_label == self.app_label or
            obj2._meta.app_label == self.app_label):
            return True
        return None

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        if app_label == self.app_label:
            return db == 'personal_ranking'
        return None