# Troubleshooting Guide

## Common Issues and Solutions

### 404 Error: `/hosted_backend/credentials/active/`

**Symptom**: Console shows `GET http://127.0.0.1:8000/hosted_backend/credentials/active/ 404 (Not Found)`

**Cause**: This is normal when no credentials have been saved yet.

**Solution**: 
1. This error is expected on first load
2. The frontend handles this gracefully by returning `null`
3. Once you save credentials, this error will disappear

### 400 Error: POST `/hosted_backend/credentials/`

**Symptom**: Console shows `POST http://127.0.0.1:8000/hosted_backend/credentials/ 400 (Bad Request)`

**Causes**:
1. Missing CSRF token
2. Invalid data format
3. Duplicate username

**Solutions**:

#### 1. CSRF Token Issue
The service is configured to include CSRF tokens automatically. Make sure:
- Django server is running: `python manage.py runserver`
- Browser has cookies enabled
- You're accessing via the correct origin (http://127.0.0.1 or http://localhost)

#### 2. Invalid Data Format
The POST requires:
```json
{
  "username": "your-username",
  "hosted_url": "https://your-hosted-backend.com",
  "is_active": true
}
```

#### 3. Duplicate Username
If you see "username already exists", delete the old one:
```bash
python manage.py shell -c "from hosted_backend_config.models import HostedBackendCredentials; HostedBackendCredentials.objects.all().delete()"
```

### Backend Not Running

**Symptom**: `ERR_CONNECTION_REFUSED` or cannot connect

**Solution**:
```bash
cd c:\Trader_Companion
python manage.py runserver
```

### CORS Errors

**Symptom**: `CORS policy: No 'Access-Control-Allow-Origin' header`

**Solution**: Already configured in `settings.py`:
- `CORS_ALLOW_ALL_ORIGINS = True`
- `CORS_ALLOW_CREDENTIALS = True`

If still having issues, restart Django server.

### Migration Issues

**Symptom**: `no such table: hosted_backend_credentials`

**Solution**:
```bash
python manage.py makemigrations hosted_backend_config
python manage.py migrate hosted_backend_config --database=hosted_backend_db
```

### Frontend Not Loading Component

**Symptom**: "Hosted Backend" tab doesn't appear

**Solution**:
1. Check that `HostedBackendConfig` component is imported
2. Verify the tab is added to `PriceAlertsPage.tsx`
3. Rebuild frontend: `npm run build` (if needed)

### Axios Type Errors

**Symptom**: TypeScript errors about `Untyped function calls`

**Solution**: These are warnings and don't affect functionality. The code works correctly despite these TypeScript warnings.

## Verification Steps

### 1. Test Backend Endpoints

```powershell
# List all credentials
curl http://127.0.0.1:8000/hosted_backend/credentials/

# Get active credentials (will be 404 if none exist - this is OK)
curl http://127.0.0.1:8000/hosted_backend/credentials/active/

# Create test credentials
$body = @{username='test'; hosted_url='https://example.com'; is_active=$true} | ConvertTo-Json
Invoke-WebRequest -Uri 'http://127.0.0.1:8000/hosted_backend/credentials/' -Method POST -Body $body -ContentType 'application/json'

# Get active again (should work now)
curl http://127.0.0.1:8000/hosted_backend/credentials/active/

# Clean up
python manage.py shell -c "from hosted_backend_config.models import HostedBackendCredentials; HostedBackendCredentials.objects.all().delete()"
```

### 2. Check Database

```bash
python manage.py shell -c "from hosted_backend_config.models import HostedBackendCredentials; print(f'Total credentials: {HostedBackendCredentials.objects.count()}')"
```

### 3. Verify URLs

```bash
python manage.py show_urls | Select-String "hosted"
```

## Configuration Checklist

- [x] `hosted_backend_config` in `INSTALLED_APPS`
- [x] `hosted_backend_db` in `DATABASES`
- [x] `HostedBackendRouter` in `DATABASE_ROUTERS`
- [x] `path('hosted_backend/', include('hosted_backend_config.urls'))` in main urls.py
- [x] Migrations run: `python manage.py migrate hosted_backend_config --database=hosted_backend_db`
- [x] CORS enabled: `CORS_ALLOW_ALL_ORIGINS = True`
- [x] REST Framework configured with no auth required

## Still Having Issues?

1. **Restart Django server**:
   ```bash
   # Stop with Ctrl+C
   python manage.py runserver
   ```

2. **Clear browser cache** and reload the page

3. **Check Django console** for error messages

4. **Check browser console** (F12) for detailed error messages

5. **Verify all files are saved** and no syntax errors exist

## Expected Behavior

### First Load (No Credentials)
- ✅ GET `/credentials/active/` returns 404 - **This is normal**
- ✅ Frontend shows configuration form
- ✅ No error dialogs shown to user

### After Saving Credentials
- ✅ GET `/credentials/active/` returns 200 with data
- ✅ POST `/credentials/` returns 201 with created object
- ✅ Auto-sync starts automatically
- ✅ "Last synced" timestamp updates

### When Testing Connection
- ✅ Hosted URL is validated
- ✅ Connection test shows success/failure
- ✅ Clear feedback to user

## Quick Reset

If everything is broken, reset the feature:

```bash
# Delete all credentials
python manage.py shell -c "from hosted_backend_config.models import HostedBackendCredentials; HostedBackendCredentials.objects.all().delete()"

# Restart server
# Ctrl+C to stop
python manage.py runserver

# Reload browser page (Ctrl+Shift+R for hard reload)
```

## Contact

If issues persist, check:
- `IMPLEMENTATION_SUMMARY.md` - Technical details
- `HOSTED_BACKEND_SETUP.md` - Deployment guide
- `QUICK_START.md` - Setup checklist
