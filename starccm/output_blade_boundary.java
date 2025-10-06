// Simcenter STAR-CCM+ macro: output_blade_boundary.java
// Written by Simcenter STAR-CCM+ 19.02.009
package macro;

import java.util.*;

import star.common.*;
import star.base.neo.*;

public class output_blade_boundary extends StarMacro {

  public void execute() {
    execute0();
  }

  private void execute0() {

    Simulation simulation_0 = 
      getActiveSimulation();

    XyzInternalTable xyzInternalTable_0 = 
      simulation_0.getTableManager().create("star.common.XyzInternalTable");

    PrimitiveFieldFunction primitiveFieldFunction_0 = 
      ((PrimitiveFieldFunction) simulation_0.getFieldFunctionManager().getFunction("MachNumber"));

    PrimitiveFieldFunction primitiveFieldFunction_1 = 
      ((PrimitiveFieldFunction) simulation_0.getFieldFunctionManager().getFunction("Pressure"));

    PrimitiveFieldFunction primitiveFieldFunction_2 = 
      ((PrimitiveFieldFunction) simulation_0.getFieldFunctionManager().getFunction("Velocity"));

    VectorComponentFieldFunction vectorComponentFieldFunction_0 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_2.getComponentFunction(0));

    VectorComponentFieldFunction vectorComponentFieldFunction_1 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_2.getComponentFunction(1));

    VectorComponentFieldFunction vectorComponentFieldFunction_2 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_2.getComponentFunction(2));

    xyzInternalTable_0.setFieldFunctions(new ArrayList<>(Arrays.<FieldFunction>asList(primitiveFieldFunction_0, primitiveFieldFunction_1, vectorComponentFieldFunction_0, vectorComponentFieldFunction_1, vectorComponentFieldFunction_2)));

    xyzInternalTable_0.getParts().setQuery(null);

    Region region_0 = 
      simulation_0.getRegionManager().getRegion("BLADE");

    Boundary boundary_0 = 
      region_0.getBoundaryManager().getBoundary("FACES");

    xyzInternalTable_0.getParts().setObjects(boundary_0);

    xyzInternalTable_0.setPresentationName("XYZ blade boundary");

    xyzInternalTable_0.extract();

    xyzInternalTable_0.export("C:\\Users\\86176\\Desktop\\python\\AICFD\\starccm\\XYZ blade boundary.csv", ",");
  }
}
