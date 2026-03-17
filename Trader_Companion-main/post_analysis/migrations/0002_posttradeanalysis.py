from django.db import migrations, models
import django.core.validators
import django.contrib.auth.models
import django.db.models.deletion

class Migration(migrations.Migration):

    dependencies = [
        ('auth', '0012_alter_user_first_name_max_length'),
        ('post_analysis', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='PostTradeAnalysis',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('trade_id', models.IntegerField(db_index=True)),
                ('title', models.CharField(blank=True, max_length=200)),
                ('notes', models.TextField(blank=True)),
                ('image', models.ImageField(blank=True, null=True, upload_to='post_analysis_images/', validators=[django.core.validators.FileExtensionValidator(['png', 'jpg', 'jpeg', 'gif'])])),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('created_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='auth.user')),
            ],
            options={
                'verbose_name': 'Post Trade Analysis',
                'verbose_name_plural': 'Post Trade Analyses',
                'ordering': ['-updated_at'],
            },
        ),
        migrations.AddIndex(
            model_name='posttradeanalysis',
            index=models.Index(fields=['trade_id'], name='post_analys_trade_i_abc123'),
        ),
    ]
