# Azure ML Authentication Guide

This guide explains how to authenticate to Azure ML from different environments (local, Colab, Kaggle).

## Authentication Methods

### 1. Local Environment (Recommended: Azure CLI)

For local development, the easiest method is to use Azure CLI:

```bash
# Login to Azure
az login

# Verify your subscription
az account show

# The code will automatically use DefaultAzureCredential which includes Azure CLI
```

**No additional configuration needed** - the code will automatically detect your Azure CLI authentication.

### 2. Colab/Kaggle (Required: Service Principal)

Colab and Kaggle environments cannot use interactive login or managed identity, so you **must** use Service Principal credentials.

## Creating a Service Principal

### Option A: If You Have Azure AD Admin Rights

If you have permissions to create service principals in Azure AD:

```bash
# For Windows/Git Bash: Disable path conversion to prevent /subscriptions/ from being converted
MSYS_NO_PATHCONV=1 az ad sp create-for-rbac \
  --name "resume-ner-colab-sp" \
  --role contributor \
  --scopes /subscriptions/<your-subscription-id>/resourceGroups/<your-resource-group>
```

**Alternative for Windows/Git Bash:** Use PowerShell instead:
```powershell
az ad sp create-for-rbac `
  --name "resume-ner-colab-sp" `
  --role contributor `
  --scopes /subscriptions/<your-subscription-id>/resourceGroups/<your-resource-group>
```

**Alternative 2:** Create SP without role assignment, then assign role separately:
```bash
# Step 1: Create service principal (skip role assignment)
az ad sp create-for-rbac \
  --name "resume-ner-colab-sp" \
  --skip-assignment

# Step 2: Get the appId from output, then assign role
MSYS_NO_PATHCONV=1 az role assignment create \
  --assignee <appId-from-step-1> \
  --role contributor \
  --scope /subscriptions/<your-subscription-id>/resourceGroups/<your-resource-group>
```

This will output:

```json
{
  "appId": "xxxx-xxxx-xxxx",      # Use as AZURE_CLIENT_ID
  "password": "xxxx-xxxx",         # Use as AZURE_CLIENT_SECRET
  "tenant": "xxxx-xxxx"            # Use as AZURE_TENANT_ID
}
```

### Option B: If You Don't Have Azure AD Admin Rights

Ask your Azure administrator to create a Service Principal with these requirements:

1. **Application Name**: `resume-ner-colab-sp` (or any name)
2. **Role**: `Contributor` (or `Reader` if you only need to read MLflow data)
3. **Scope**: Your resource group: `/subscriptions/<subscription-id>/resourceGroups/<resource-group>`

The administrator can create it via:

- Azure Portal: Azure Active Directory > App registrations > New registration
- Azure CLI: `az ad sp create-for-rbac` (as shown above)
- PowerShell: `New-AzADServicePrincipal`

Once created, ask them to provide:

- **Application (client) ID** → `AZURE_CLIENT_ID`
- **Client secret value** → `AZURE_CLIENT_SECRET`
- **Directory (tenant) ID** → `AZURE_TENANT_ID`

### Option C: Use Existing Service Principal

If you already have a Service Principal, you can use its credentials. You may need to:

1. Create a new client secret if you don't have one
2. Ensure it has Contributor role on your resource group

## Configuring Credentials

### For Colab/Kaggle

Add Service Principal credentials to your `config.env` file:

```bash
# Required for Azure ML connection
AZURE_SUBSCRIPTION_ID="a23fa87c-802c-4fdf-9e59-e3d7969bcf31"
AZURE_RESOURCE_GROUP="resume_ner_2025-12-14-13-17-35"
AZURE_LOCATION="southeastasia"

# Required for Colab/Kaggle authentication
AZURE_CLIENT_ID="<your-service-principal-client-id>"
AZURE_CLIENT_SECRET="<your-service-principal-client-secret>"
AZURE_TENANT_ID="<your-azure-tenant-id>"
```

**Important**:

- Upload `config.env` to Colab/Kaggle (don't commit secrets to git!)
- The file should be in the project root directory
- Make sure `.env` files are in `.gitignore`

### For Local Environment

You have two options:

**Option 1: Azure CLI (Recommended)**

- Just run `az login`
- No need to add Service Principal credentials to `config.env`

**Option 2: Service Principal**

- Add Service Principal credentials to `config.env` (same as Colab/Kaggle)
- Useful if you can't use Azure CLI or need non-interactive authentication

## Verifying Authentication

After setting up credentials, the code will:

1. **Load credentials** from `config.env` if environment variables aren't set
2. **Detect platform** (Colab/Kaggle vs Local)
3. **Choose authentication method**:
   - Colab/Kaggle: Uses `ClientSecretCredential` (Service Principal)
   - Local: Uses `DefaultAzureCredential` (tries Azure CLI, then Service Principal env vars)

## Troubleshooting

### "Insufficient privileges to complete the operation"

**Solution**: Ask your Azure administrator to create the Service Principal for you (see Option B above).

### "DefaultAzureCredential failed to retrieve a token"

**In Colab/Kaggle**:

- Ensure Service Principal credentials are in `config.env`
- Verify the credentials are correct
- Check that the Service Principal has Contributor role on the resource group

**In Local**:

- Try `az login` first
- Or add Service Principal credentials to `config.env`

### "Environment variables are not fully configured"

This means Service Principal credentials are missing. Add them to `config.env`:

- `AZURE_CLIENT_ID`
- `AZURE_CLIENT_SECRET`
- `AZURE_TENANT_ID`

### Authentication works but connection fails

Check:

1. Subscription ID is correct
2. Resource Group name is correct
3. Workspace name matches `config/mlflow.yaml` (default: `resume-ner-ws`)
4. Service Principal has Contributor role on the resource group

## Security Best Practices

1. **Never commit secrets to git** - Keep `config.env` in `.gitignore`
2. **Use least privilege** - Service Principal should only have Contributor (not Owner) role
3. **Rotate secrets regularly** - Update `AZURE_CLIENT_SECRET` periodically
4. **Scope permissions** - Limit Service Principal to specific resource groups, not entire subscription
5. **Use separate Service Principals** - Different ones for different environments (dev, prod)

## Alternative: Use Local MLflow Tracking

If you can't set up Azure ML authentication, the code will automatically fall back to local SQLite tracking:

- **Colab**: `/content/drive/MyDrive/resume-ner-mlflow/mlflow.db`
- **Kaggle**: `/kaggle/working/mlflow.db`
- **Local**: `./mlruns/mlflow.db`

This works fine for development, but you won't have centralized tracking across runs.
