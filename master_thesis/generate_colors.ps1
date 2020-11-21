
For ($i=0; $i -le 40; $i++) {
    $r = Get-Random -Minimum -0 -Maximum 255
    $g = Get-Random -Minimum -0 -Maximum 255
    $b = Get-Random -Minimum -0 -Maximum 255

    # "C2": [74,179,255],
    '"' + "OUTER_$i" + '": ' + "[$r,$g,$b],"
}