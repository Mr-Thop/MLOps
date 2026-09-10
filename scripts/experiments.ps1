$ErrorActionPreference = "Stop"

$experiments = @(
    @{ name = "exp01-lr-0001";        overrides = @("model_training.learning_rate=0.0001") }
    @{ name = "exp02-lr-0005";        overrides = @("model_training.learning_rate=0.0005") }
    @{ name = "exp03-lr-baseline";    overrides = @("model_training.learning_rate=0.001") }
    @{ name = "exp04-lr-005";         overrides = @("model_training.learning_rate=0.005") }
    @{ name = "exp05-lr-01";          overrides = @("model_training.learning_rate=0.01") }
    @{ name = "exp06-batch-32";       overrides = @("model_training.batch_size=32") }
    @{ name = "exp07-batch-128";      overrides = @("model_training.batch_size=128") }
    @{ name = "exp08-batch-256";      overrides = @("model_training.batch_size=256") }
    @{ name = "exp09-dropout-01";     overrides = @("model_training.dropout_rate=0.1") }
    @{ name = "exp10-dropout-05";     overrides = @("model_training.dropout_rate=0.5") }
    @{ name = "exp11-optimizer-sgd";  overrides = @("model_training.optimizer=sgd") }
    @{ name = "exp12-optimizer-rmsprop"; overrides = @("model_training.optimizer=rmsprop") }
    @{ name = "exp13-hidden-512-256"; overrides = @("model_training.hidden_units=[512,256]") }
    @{ name = "exp14-hidden-128-64";  overrides = @("model_training.hidden_units=[128,64]") }
    @{ name = "exp15-hidden-3layer";  overrides = @("model_training.hidden_units=[512,256,128]") }
    @{ name = "exp16-epochs-10";      overrides = @("model_training.epochs=10") }
    @{ name = "exp17-epochs-40";      overrides = @("model_training.epochs=40") }
    @{ name = "exp18-pca-50";         overrides = @("feature_selection.pca_n_components=50") }
    @{ name = "exp19-pca-200";        overrides = @("feature_selection.pca_n_components=200") }
    @{ name = "exp20-combo-best-guess"; overrides = @(
            "model_training.learning_rate=0.0005",
            "model_training.batch_size=128",
            "model_training.dropout_rate=0.4",
            "model_training.hidden_units=[512,256]"
        )
    }
)

$counter = 0
foreach ($exp in $experiments) {
    $counter++
    $name = $exp.name
    $flags = @()
    foreach ($ov in $exp.overrides) {
        $flags += "-S"
        $flags += $ov
    }

    Write-Host ""
    Write-Host "=============================================================" -ForegroundColor Cyan
    Write-Host " [$counter/20] Running experiment: $name" -ForegroundColor Cyan
    Write-Host " Overrides: $($exp.overrides -join ', ')" -ForegroundColor Cyan
    Write-Host "=============================================================" -ForegroundColor Cyan

    dvc exp run -n $name @flags

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Experiment '$name' failed (exit code $LASTEXITCODE)." -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "All 20 experiments finished. Run 'dvc exp show' to compare them," -ForegroundColor Green
Write-Host "or '.\scripts\export_results.ps1' to export the comparison to CSV." -ForegroundColor Green