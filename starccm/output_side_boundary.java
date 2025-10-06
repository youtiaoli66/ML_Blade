// Simcenter STAR-CCM+ macro: output_side_boundary.java
// Written by Simcenter STAR-CCM+ 19.02.009
package macro;

import java.util.*;

import star.common.*;
import star.base.neo.*;

public class output_side_boundary extends StarMacro {

  public void execute() {
    execute0();
  }

  private void execute0() {

    Simulation simulation_0 = 
      getActiveSimulation();

    XyzInternalTable xyzInternalTable_1 = 
      simulation_0.getTableManager().create("star.common.XyzInternalTable");

    xyzInternalTable_1.setPresentationName("XYZ side boundary");

    PrimitiveFieldFunction primitiveFieldFunction_0 = 
      ((PrimitiveFieldFunction) simulation_0.getFieldFunctionManager().getFunction("MachNumber"));

    xyzInternalTable_1.setFieldFunctions(new ArrayList<>(Arrays.<FieldFunction>asList(primitiveFieldFunction_0)));

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

    xyzInternalTable_1.setFieldFunctions(new ArrayList<>(Arrays.<FieldFunction>asList(primitiveFieldFunction_0, primitiveFieldFunction_1, vectorComponentFieldFunction_0, vectorComponentFieldFunction_1, vectorComponentFieldFunction_2)));

    xyzInternalTable_1.getParts().setQuery(null);

    Region region_0 = 
      simulation_0.getRegionManager().getRegion("BLADE");

    Boundary boundary_1 = 
      region_0.getBoundaryManager().getBoundary("INLET");

    InterfaceBoundary interfaceBoundary_0 = 
      ((InterfaceBoundary) region_0.getBoundaryManager().getBoundary("INLET [BLADE/HEAD]"));

    Boundary boundary_2 = 
      region_0.getBoundaryManager().getBoundary("OUTLET");

    InterfaceBoundary interfaceBoundary_1 = 
      ((InterfaceBoundary) region_0.getBoundaryManager().getBoundary("OUTLET [BLADE/TAIL]"));

    Boundary boundary_3 = 
      region_0.getBoundaryManager().getBoundary("PER1");

    InterfaceBoundary interfaceBoundary_2 = 
      ((InterfaceBoundary) region_0.getBoundaryManager().getBoundary("PER1 [BLADE/BLADE]"));

    Boundary boundary_4 = 
      region_0.getBoundaryManager().getBoundary("PER2");

    InterfaceBoundary interfaceBoundary_3 = 
      ((InterfaceBoundary) region_0.getBoundaryManager().getBoundary("PER2 [BLADE/BLADE]"));

    xyzInternalTable_1.getParts().setObjects(boundary_1, interfaceBoundary_0, boundary_2, interfaceBoundary_1, boundary_3, interfaceBoundary_2, boundary_4, interfaceBoundary_3);

    xyzInternalTable_1.extract();

    xyzInternalTable_1.export("C:\\Users\\86176\\Desktop\\python\\AICFD\\starccm\\XYZ side boundary.csv", ",");
  }
}
