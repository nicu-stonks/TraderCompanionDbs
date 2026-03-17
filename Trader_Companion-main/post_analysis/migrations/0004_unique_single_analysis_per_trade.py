from django.db import migrations, models


def dedupe(apps, schema_editor):
    PostTradeAnalysis = apps.get_model('post_analysis', 'PostTradeAnalysis')
    # For each trade_id keep the most recently updated row, delete the rest
    from collections import defaultdict
    buckets = defaultdict(list)
    for obj in PostTradeAnalysis.objects.all().order_by('-updated_at'):
        buckets[obj.trade_id].append(obj)
    to_delete = []
    for trade_id, items in buckets.items():
        # items already sorted newest first; skip index 0
        for stale in items[1:]:
            to_delete.append(stale.id)
    if to_delete:
        PostTradeAnalysis.objects.filter(id__in=to_delete).delete()


class Migration(migrations.Migration):

    dependencies = [
        ('post_analysis', '0003_rename_post_analys_trade_i_abc123_post_analys_trade_i_9bd628_idx_and_more'),
    ]

    operations = [
        migrations.RunPython(dedupe, migrations.RunPython.noop),
        migrations.AddConstraint(
            model_name='posttradeanalysis',
            constraint=models.UniqueConstraint(fields=('trade_id',), name='uniq_post_analysis_trade'),
        ),
    ]
