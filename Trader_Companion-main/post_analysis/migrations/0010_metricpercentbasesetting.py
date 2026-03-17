from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('post_analysis', '0009_metricgradechecksetting'),
    ]

    operations = [
        migrations.CreateModel(
            name='MetricPercentBaseSetting',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('metric_id', models.IntegerField(unique=True)),
                ('use_total_trades', models.BooleanField(default=False)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'ordering': ['metric_id'],
            },
        ),
    ]
