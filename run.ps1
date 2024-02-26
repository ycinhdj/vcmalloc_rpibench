$exePath = ".\vcmalloc_rpibench.exe"

$scenarios = @(
    "knn_m",
    "knn_vcm",
    "knn_vcma"
)

$start = 10
$end = 17
$iterations = 5

foreach ($scenario in $scenarios) {
    for ($i = $start; $i -le $end; $i++) {
        $power = [math]::Pow(2, $i)
        for ($j = 0; $j -lt $iterations; $j++) {
            Write-Host "Executing $exePath with argument $power 1024 10 for scenario $scenario"
            Start-Process -FilePath $exePath -ArgumentList "$scenario $power 1024 10" -Wait
        }
    }
}

$scenarios = @(
    "kmeans_m",
    "kmeans_vcm",
    "kmeans_vcma"
)

$start = 10
$end = 17
$iterations = 1

foreach ($scenario in $scenarios) {
    for ($i = $start; $i -le $end; $i++) {
        $power = [math]::Pow(2, $i)
        for ($k = 0; $k -lt $iterations; $k++) {
            Write-Host "Executing $exePath with argument $power 1024 100 10 for scenario $scenario"
            Start-Process -FilePath $exePath -ArgumentList "$scenario $power 1024 100 10"  -Wait
        }
        
    }
}

$scenarios = @(
    "matmult_m",
    "matmult_vcm",
    "matmult_vcma"
)

$start = 2
$end = 8
$iterations = 1

foreach ($scenario in $scenarios) {
    for ($i = $start; $i -le $end; $i++) {
        $power = [math]::Pow(2, $i)
        for ($l = 0; $l -lt $iterations; $l++) {
            Write-Host "Executing $exePath with argument $power $power $power for scenario $scenario"
            Start-Process -FilePath $exePath -ArgumentList "$scenario $power $power $power"  -Wait
        }
    }
}
