from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('post_analysis', '0007_alter_posttradeanalysis_options_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='MetricOptionRecommendation',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('metric_id', models.IntegerField()),
                ('option_id', models.IntegerField()),
                ('recommended_pct', models.DecimalField(decimal_places=2, max_digits=5)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'ordering': ['metric_id', 'option_id'],
                'unique_together': {('metric_id', 'option_id')},
            },
        ),
    ]
