use nalgebra_glm::{Mat4, Vec4};

pub struct FrustrumCuller {
    nx_x: f32,
    nx_y: f32,
    nx_z: f32,
    nx_w: f32,
    px_x: f32,
    px_y: f32,
    px_z: f32,
    px_w: f32,
    ny_x: f32,
    ny_y: f32,
    ny_z: f32,
    ny_w: f32,
    py_x: f32,
    py_y: f32,
    py_z: f32,
    py_w: f32,
    nz_x: f32,
    nz_y: f32,
    nz_z: f32,
    nz_w: f32,
    pz_x: f32,
    pz_y: f32,
    pz_z: f32,
    pz_w: f32,
}

impl FrustrumCuller {
    pub fn from_matrix(m: Mat4) -> Self {
        let mut culler: Self = unsafe { std::mem::zeroed() };

        culler.nx_x = m.column(0)[3] + m.column(0)[0];
        culler.nx_y = m.column(1)[3] + m.column(1)[0];
        culler.nx_z = m.column(2)[3] + m.column(2)[0];
        culler.nx_w = m.column(3)[3] + m.column(3)[0];
        //if (allow_test_spheres) {
        let invl =
            (culler.nx_x * culler.nx_x + culler.nx_y * culler.nx_y + culler.nx_z * culler.nx_z)
                .sqrt()
                .recip();
        culler.nx_x *= invl;
        culler.nx_y *= invl;
        culler.nx_z *= invl;
        culler.nx_w *= invl;
        //}
        culler.px_x = m.column(0)[3] - m.column(0)[0];
        culler.px_y = m.column(1)[3] - m.column(1)[0];
        culler.px_z = m.column(2)[3] - m.column(2)[0];
        culler.px_w = m.column(3)[3] - m.column(3)[0];
        //if (allow_test_spheres) {
        let invl =
            (culler.px_x * culler.px_x + culler.px_y * culler.px_y + culler.px_z * culler.px_z)
                .sqrt()
                .recip();
        culler.px_x *= invl;
        culler.px_y *= invl;
        culler.px_z *= invl;
        culler.px_w *= invl;
        //}
        culler.ny_x = m.column(0)[3] + m.column(0)[1];
        culler.ny_y = m.column(1)[3] + m.column(1)[1];
        culler.ny_z = m.column(2)[3] + m.column(2)[1];
        culler.ny_w = m.column(3)[3] + m.column(3)[1];
        //if (allow_test_spheres) {
        let invl =
            (culler.ny_x * culler.ny_x + culler.ny_y * culler.ny_y + culler.ny_z * culler.ny_z)
                .sqrt()
                .recip();
        culler.ny_x *= invl;
        culler.ny_y *= invl;
        culler.ny_z *= invl;
        culler.ny_w *= invl;
        //}
        culler.py_x = m.column(0)[3] - m.column(0)[1];
        culler.py_y = m.column(1)[3] - m.column(1)[1];
        culler.py_z = m.column(2)[3] - m.column(2)[1];
        culler.py_w = m.column(3)[3] - m.column(3)[1];
        //if (allow_test_spheres) {
        let invl =
            (culler.py_x * culler.py_x + culler.py_y * culler.py_y + culler.py_z * culler.py_z)
                .sqrt()
                .recip();
        culler.py_x *= invl;
        culler.py_y *= invl;
        culler.py_z *= invl;
        culler.py_w *= invl;
        //}
        culler.nz_x = m.column(0)[3] + m.column(0)[2];
        culler.nz_y = m.column(1)[3] + m.column(1)[2];
        culler.nz_z = m.column(2)[3] + m.column(2)[2];
        culler.nz_w = m.column(3)[3] + m.column(3)[2];
        //if (allow_test_spheres) {
        let invl =
            (culler.nz_x * culler.nz_x + culler.nz_y * culler.nz_y + culler.nz_z * culler.nz_z)
                .sqrt()
                .recip();
        culler.nz_x *= invl;
        culler.nz_y *= invl;
        culler.nz_z *= invl;
        culler.nz_w *= invl;
        //}
        culler.pz_x = m.column(0)[3] - m.column(0)[2];
        culler.pz_y = m.column(1)[3] - m.column(1)[2];
        culler.pz_z = m.column(2)[3] - m.column(2)[2];
        culler.pz_w = m.column(3)[3] - m.column(3)[2];
        //if (allow_test_spheres) {
        let invl =
            (culler.pz_x * culler.pz_x + culler.pz_y * culler.pz_y + culler.pz_z * culler.pz_z)
                .sqrt()
                .recip();
        culler.pz_x *= invl;
        culler.pz_y *= invl;
        culler.pz_z *= invl;
        culler.pz_w *= invl;
        //}

        culler
    }

    /// Returns the result of testing the intersection of the frustum with a sphere, defined by a
    /// center point (`center`) and a radius (`radius`).
    ///
    /// This method will distinguish between a partial intersection and a total intersection.
    pub fn test_sphere(&self, sphere: Vec4) -> bool {
        let sphere_center = sphere.xyz();
        let sphere_radius = sphere[3];

        let mut dist;
        dist = self.nx_x * sphere_center[0]
            + self.nx_y * sphere_center.y
            + self.nx_z * sphere_center.z
            + self.nx_w;
        if dist >= -sphere_radius {
            dist = self.px_x * sphere_center[0]
                + self.px_y * sphere_center.y
                + self.px_z * sphere_center.z
                + self.px_w;
            if dist >= -sphere_radius {
                dist = self.ny_x * sphere_center[0]
                    + self.ny_y * sphere_center.y
                    + self.ny_z * sphere_center.z
                    + self.ny_w;
                if dist >= -sphere_radius {
                    dist = self.py_x * sphere_center[0]
                        + self.py_y * sphere_center.y
                        + self.py_z * sphere_center.z
                        + self.py_w;
                    if dist >= -sphere_radius {
                        dist = self.nz_x * sphere_center[0]
                            + self.nz_y * sphere_center.y
                            + self.nz_z * sphere_center.z
                            + self.nz_w;
                        if dist >= -sphere_radius {
                            dist = self.pz_x * sphere_center[0]
                                + self.pz_y * sphere_center.y
                                + self.pz_z * sphere_center.z
                                + self.pz_w;
                            if dist >= -sphere_radius {
                                return true;
                            }
                        }
                    }
                }
            }
        }

        false
    }
}
