
<style>
    table,th, tr ,td {
        border: 1px solid black;
        border-collapse: collapse;
    }
    td{
      border: 3px solid black;
      border-collapse: inherit;
    }
</style>


## TRACK EXPERIMENTS

### BACKBONE
a table of backbones with diferent losses and datasets reporting the initial and final simularity metrics between the source and target image pairs averaged across the entire dataset.




<table>
      <tr>
        <th rowspan="2" style="transform: rotate(0deg); color:dimgrey;" >Backbone</th>
        <th colspan="3">SkyData</th>
        <th colspan="3">VEDAI</th>
      </tr>
      <tr>
        <th>fir_frgb</th>        
        <th>frgb_Irgb</th>
        <th>fir_Irgb</th>
        <th>fir_frgb</th>        
        <th>frgb_Irgb</th>
        <th>fir_Irgb</th>
      </tr>
      <tr>
        <th rowspan="2">mse_pixel</th>
        <td>0.056535</td>
        <td>0.043476</td>
        <td>0.015780</td>
        <td>0.021924</td>
        <td>0.014260</td>
        <td>0.018490</td>
      </tr>
      <tr>
        <td>0.048425</td>
        <td>0.042931</td>
        <td>0.007486</td>
        <td>0.017489</td>
        <td>0.014638</td>
        <td>0.013117</td>
      </tr>
      <tr>
        <th rowspan="2">mae_pixel</th>
        <td>0.324681</td>
        <td>0.050360</td>
        <td>0.085442</td>
        <td>0.166066</td>
        <td>0.046108</td>
        <td>0.112023</td>
      </tr>
      <tr>
        <td>0.327348</td>
        <td>0.035603</td>
        <td>0.059121</td>
        <td>0.168939</td>
        <td>0.075681</td>
        <td>0.087706</td>
      </tr>
      <tr>
        <th rowspan="2">sse_pixel</th>
        <td>40652.160</td>
        <td>15079.448</td>
        <td>13122.528</td>
        <td>15341.776</td>
        <td>4668.434</td>
        <td>10046.029</td>
      </tr>
      <tr>
        <td>28144.173</td>
        <td>14316.845</td>
        <td>6967.163</td>
        <td>4903.374</td>
        <td>10721.511</td>
        <td>7048.494</td>
      </tr>
      <tr>
        <th rowspan="2">ssim_pixel</th>
        <td>0.766243</td>
        <td>0.057376</td>
        <td>0.394650</td>
        <td>0.190374</td>
        <td>0.084341</td>
        <td>0.298926</td>
      </tr>
      <tr>
        <td>0.705532</td>
        <td>0.087164</td>
        <td>0.273515</td>
        <td>0.152903</td>
        <td>0.081297</td>
        <td>0.240657</td>
      </tr>
    </table>

### Regression head
for each dataset there will be regression head trained on different backbones with different loss functions (l2_corner_loss). we will report the average corner error for each loss function.

<table>
      <tr>
        <th rowspan="2" style="transform: rotate(0deg); color:dimgrey;" >Backbone</th>
        <th colspan="2">SkyData</th>
        <th colspan="2">VEDAI</th>
      </tr>
      <tr>
        <th>l2_corner_loss</th>        
        <th>l2_homography</th>
        <th>l2_corner_loss</th>        
        <th>l2_homography</th>
      </tr>
      <tr>
        <th rowspan="1">mse_pixel</th>
        <td>18.590597</td>
        <td>142.90131</td>
        <td>19.140575</td>
        <td> </td>
      </tr>
      <tr>
        <th rowspan="1">mae_pixel</th>
        <td>18.499010 </td>
        <td>35.835938 </td>
        <td>18.533229 </td>
        <td> </td>
      </tr>
      <tr>
        <th rowspan="1">sse_pixel</th>
        <td>19.071471</td>
        <td>42.270582</td>
        <td>20.671191</td>
        <td> </td>
      </tr>
      <tr>
        <th rowspan="1">ssim_pixel</th>
        <td>18.505228</td>
        <td>20.105120</td>
        <td>19.114550</td>
        <td> </td>
      </tr>
    </table>








<!-- ### regression head
for each dataset there will be regression head trained on different backbones with different regression losses

| Backbone  | R_loss | SkyData    |VEDAI  |
|-----------|-----------------|-----------|-------|
| mse_pixel | l2_corners_loss | &check;       | &check;   | 
| mae_pixel | l2_corners_loss | &check;       | &check;   | 
| sse_pixel | l2_corners_loss | &check;       | &check;   | 
| ssim_pixel| l2_corners_loss | &check;       | &check;   |
***

| Backbone  | R_loss | SkyData    |VEDAI  |
|-----------|-----------------|-----------|-------|
| mse_pixel | l2_homography_loss | -      | -   | 
| mae_pixel | l2_homography_loss | -      | -   |
| sse_pixel | l2_homography_loss | -      | -   |
| ssim_pixel| l2_homography_loss | -      | -   |  -->
<!-- 

<table>
      <tr>
        <th></th>
        <th colspan="3">SkyData</th>
        <th colspan="3">VEDAI</th>
      </tr>
      <tr>
        <th></th>
        <th>fir_frgb</th>        
        <th>frgb_Irgb</th>
        <th>fir_Irgb</th>
        <th>fir_frgb</th>        
        <th>frgb_Irgb</th>
        <th>fir_Irgb</th>
      </tr>
      <tr>
        <th rowspan="2">mse_pixel</th>
        <td> </td>
        <td> </td>
        <td> </td>
        <td> </td>
        <td> </td>
        <td> </td>
      </tr>
      <tr>
        <td> </td>
        <td> </td>
        <td> </td>
        <td> </td>
        <td> </td>
        <td> </td>
      </tr>
      <tr>
        <th rowspan="2">mae_pixel</th>
        <td> </td>
        <td> </td>
        <td> </td>
        <td> </td>
        <td> </td>
        <td> </td>
      </tr>
      <tr>
        <td> </td>
        <td> </td>
        <td> </td>
        <td> </td>
        <td> </td>
        <td> </td>
      </tr>
      <tr>
        <th rowspan="2">sse_pixel</th>
        <td> </td>
        <td> </td>
        <td> </td>
        <td> </td>
        <td> </td>
        <td> </td>
      </tr>
      <tr>
        <td> </td>
        <td> </td>
        <td> </td>
        <td> </td>
        <td> </td>
        <td> </td>
      </tr>
      <tr>
        <th rowspan="2">ssim_pixel</th>
        <td> </td>
        <td> </td>
        <td> </td>
        <td> </td>
        <td> </td>
        <td> </td>
      </tr>
      <tr>
        <td> </td>
        <td> </td>
        <td> </td>
        <td> </td>
        <td> </td>
        <td> </td>
      </tr>
    </table>
 -->