// Generates the racing scenes of the macOS Flightmare renderer from Unity
// primitives and procedural textures (no third-party assets):
//   scene_id 0  warehouse  56 x 40 x 12 m steel hall: pallet racks, crates, forklift
//   scene_id 2  garage     56 x 40 x 7 m concrete parking garage: columns, parked cars
//   scene_id 4  arena      300 m checkered ground, sky
// (flightlib accepts ids < 5: the halls take the slots of the non-public
// upstream Warehouse and Mountain scenes)
// The halls are centered on the bundled tracks (acmpc/tracks) and keep their
// volume free: x -9..13.5 m, y -10..10.5 m, up to 6 m (ROS frame, track at offset 0).
// Gates come from the static_obstacles.csv that the race overlay writes
// (acmpc/race_track.py).
// Copied into <flightmare_unity>/Assets/Editor/ by scripts/build_renderer.sh;
// BuildMacOS.Build calls RaceScenes.Ensure() and adds the scenes to the build;
// build_renderer.sh registers them in sceneSchedule.cs.
using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;
using Object = UnityEngine.Object;
using Random = System.Random;

public static class RaceScenes
{
    public const string ArenaPath = "Assets/Environments/Arena/Arena.unity";
    public const string WarehousePath = "Assets/Environments/RaceWarehouse/RaceWarehouse.unity";
    public const string GaragePath = "Assets/Environments/RaceGarage/RaceGarage.unity";
    public static readonly string[] ScenePaths = { ArenaPath, WarehousePath, GaragePath };

    // Hall center and size (ROS frame: x forward, y left, z up)
    const float Cx = 2.25f, Cy = 0.25f, L = 56f, W = 40f;
    // Kept free for the track
    const float KeepX0 = -9f, KeepX1 = 13.5f, KeepY0 = -10f, KeepY1 = 10.5f;

    static string dir;
    static Transform root;

    public static void Ensure()
    {
        BuildArena();
        BuildWarehouse();
        BuildGarage();
        AssetDatabase.SaveAssets();
    }

    // ---- scenes -----------------------------------------------------------

    static void BuildArena()
    {
        Scene scene = Begin(ArenaPath, keepSun: true);
        Material ground = Mat("Ground", Color.white,
            Tex("checker", 256, (x, y) => ((x / 128) + (y / 128)) % 2 == 0
                ? new Color(0.62f, 0.64f, 0.66f) : new Color(0.42f, 0.44f, 0.46f)), 0.1f);
        // 300 m, 2 m per texture repeat -> 1 m squares
        ground.mainTextureScale = new Vector2(150, 150);
        Box("Ground", 0, 0, -0.05f, 300, 300, 0.1f, ground);
        RenderSettings.ambientMode = AmbientMode.Skybox;
        End(scene, ArenaPath);
    }

    static void BuildWarehouse()
    {
        Scene scene = Begin(WarehousePath, keepSun: false);
        var rng = new Random(5);
        const float H = 12f;

        Material floor = Mat("Floor", new Color(0.78f, 0.78f, 0.76f), Concrete("concrete", 0.55f), 0.25f);
        floor.mainTextureScale = new Vector2(L / 4, W / 4);
        Material wallLong = Mat("WallLong", new Color(0.72f, 0.76f, 0.8f), Corrugated("corrugated"), 0.3f);
        wallLong.mainTextureScale = new Vector2(L / 2, 1);
        Material wallShort = Mat("WallShort", new Color(0.72f, 0.76f, 0.8f), wallLong.mainTexture as Texture2D, 0.3f);
        wallShort.mainTextureScale = new Vector2(W / 2, 1);
        Material plinth = Mat("Plinth", new Color(0.45f, 0.45f, 0.45f), floor.mainTexture as Texture2D);
        Material roof = Mat("Roof", new Color(0.3f, 0.31f, 0.33f));
        Material steel = Mat("Steel", new Color(0.35f, 0.4f, 0.48f), null, 0.4f);
        Material skylight = Mat("Skylight", Color.white, null, 0.0f, new Color(0.9f, 0.95f, 1.0f) * 1.6f);
        Material rackBlue = Mat("RackBlue", new Color(0.1f, 0.25f, 0.6f), null, 0.35f);
        Material rackOrange = Mat("RackOrange", new Color(0.95f, 0.45f, 0.05f), null, 0.35f);
        Material pallet = Mat("Pallet", new Color(0.55f, 0.4f, 0.25f));
        Material yellow = Mat("Yellow", new Color(0.95f, 0.8f, 0.05f));
        Material door = Mat("Door", new Color(0.55f, 0.57f, 0.6f), Corrugated("door", 16), 0.3f);
        Material dark = Mat("Dark", new Color(0.12f, 0.12f, 0.13f));
        Material[] cardboard = {
            Mat("Box1", new Color(0.66f, 0.5f, 0.32f)), Mat("Box2", new Color(0.58f, 0.43f, 0.27f)),
            Mat("Box3", new Color(0.75f, 0.6f, 0.42f)), Mat("Wrap", new Color(0.8f, 0.86f, 0.9f), null, 0.6f),
        };

        Shell(H, floor, wallLong, wallShort, roof);
        // Concrete plinth along the walls
        Box("PlinthN", Cx, Cy + W / 2 - 0.2f, 0.6f, L, 0.1f, 1.2f, plinth);
        Box("PlinthS", Cx, Cy - W / 2 + 0.2f, 0.6f, L, 0.1f, 1.2f, plinth);
        Box("PlinthW", Cx - L / 2 + 0.2f, Cy, 0.6f, 0.1f, W, 1.2f, plinth);
        Box("PlinthE", Cx + L / 2 - 0.2f, Cy, 0.6f, 0.1f, W, 1.2f, plinth);

        // Roof trusses, skylights and wall columns
        for (float x = Cx - L / 2 + 3.5f; x < Cx + L / 2; x += 7f)
        {
            Box("Truss", x, Cy, H - 0.6f, 0.3f, W, 0.6f, steel);
            Box("TrussChord", x, Cy, H - 2.2f, 0.15f, W, 0.15f, steel);
            for (float y = Cy - W / 2 + 4; y < Cy + W / 2 - 3; y += 4)
                Box("TrussStrut", x, y, H - 1.4f, 0.12f, 0.12f, 1.6f, steel);
            Box("ColumnN", x, Cy + W / 2 - 0.5f, H / 2, 0.4f, 0.4f, H, steel);
            Box("ColumnS", x, Cy - W / 2 + 0.5f, H / 2, 0.4f, 0.4f, H, steel);
        }
        for (float x = Cx - 21f; x <= Cx + 21f; x += 14f)
            Box("Skylight", x, Cy, H - 0.05f, 2.0f, W - 6, 0.05f, skylight);

        // Floor markings: aisle lines along the racks, walkway around the track area
        foreach (float s in new[] { -1f, 1f })
        {
            Box("Lane", Cx, Cy + s * 13.5f, 0.005f, L - 2, 0.15f, 0.01f, yellow);
        }
        FloorRect(KeepX0 + 0.5f, KeepY0 + 0.5f, KeepX1 - 0.5f, KeepY1 - 0.5f, 0.12f, yellow);

        // Pallet racks: two back-to-back rows along each long wall
        foreach (float s in new[] { -1f, 1f })
            foreach (float yRow in new[] { 18.4f, 15.2f })
                for (float x = Cx - L / 2 + 3f; x + 2.7f < Cx + L / 2 - 2f; x += 2.8f)
                    RackBay(x + 1.35f, Cy + s * yRow, rackBlue, rackOrange, pallet, cardboard, rng);

        // Loading dock: roller doors on the west wall, dock bumpers
        for (float y = Cy - 9; y <= Cy + 9; y += 6)
        {
            Box("Door", Cx - L / 2 + 0.2f, y, 2.7f, 0.08f, 4.2f, 5.0f, door);
            Box("DoorFrame", Cx - L / 2 + 0.22f, y, 5.3f, 0.1f, 4.6f, 0.3f, yellow);
            Box("Bumper", Cx - L / 2 + 0.35f, y - 1.6f, 0.6f, 0.25f, 0.3f, 0.6f, dark);
            Box("Bumper", Cx - L / 2 + 0.35f, y + 1.6f, 0.6f, 0.25f, 0.3f, 0.6f, dark);
        }

        // Crate stacks and wrapped pallets east and west of the track
        for (int i = 0; i < 14; i++)
        {
            float x = (float)(KeepX1 + 3 + rng.NextDouble() * (Cx + L / 2 - KeepX1 - 6));
            float y = (float)(Cy - 10 + rng.NextDouble() * 20);
            PalletStack(x, y, pallet, cardboard, rng);
        }
        for (int i = 0; i < 8; i++)
        {
            float x = (float)(Cx - L / 2 + 4 + rng.NextDouble() * (KeepX0 - 3 - (Cx - L / 2 + 4)));
            float y = (float)(Cy - 11 + rng.NextDouble() * 22);
            PalletStack(x, y, pallet, cardboard, rng);
        }
        Forklift(Cx + 20f, Cy - 6f, 30f, yellow, dark, steel);
        Forklift(Cx - 18f, Cy + 7f, -120f, yellow, dark, steel);

        RenderSettings.ambientMode = AmbientMode.Flat;
        RenderSettings.ambientLight = new Color(0.36f, 0.37f, 0.4f);
        for (float x = Cx - 16; x <= Cx + 16; x += 16)
            foreach (float y in new[] { Cy - 9f, Cy + 9f })
                PointLight(x, y, H - 2.5f, 26f, 1.4f, new Color(1f, 0.96f, 0.88f));
        End(scene, WarehousePath);
    }

    static void BuildGarage()
    {
        Scene scene = Begin(GaragePath, keepSun: false);
        var rng = new Random(6);
        const float H = 7f;

        Material floor = Mat("Floor", new Color(0.55f, 0.56f, 0.57f), Concrete("concrete", 0.4f), 0.35f);
        floor.mainTextureScale = new Vector2(L / 4, W / 4);
        Material wall = Mat("Wall", new Color(0.72f, 0.72f, 0.7f), Concrete("wall", 0.25f), 0.1f);
        wall.mainTextureScale = new Vector2(L / 4, 2);
        Material wallShort = Mat("WallShort", new Color(0.72f, 0.72f, 0.7f), wall.mainTexture as Texture2D, 0.1f);
        wallShort.mainTextureScale = new Vector2(W / 4, 2);
        Material ceiling = Mat("Ceiling", new Color(0.62f, 0.62f, 0.6f), wall.mainTexture as Texture2D);
        Material concrete = Mat("Column", new Color(0.7f, 0.7f, 0.68f), wall.mainTexture as Texture2D);
        Material hazard = Mat("Hazard", Color.white, Hazard("hazard"));
        Material white = Mat("Paint", new Color(0.92f, 0.92f, 0.9f));
        Material yellow = Mat("Yellow", new Color(0.95f, 0.78f, 0.05f));
        Material blue = Mat("Band", new Color(0.1f, 0.35f, 0.75f));
        Material tube = Mat("Tube", Color.white, null, 0.0f, new Color(0.85f, 0.95f, 1.0f) * 2.0f);
        Material pipe = Mat("Pipe", new Color(0.7f, 0.15f, 0.12f), null, 0.4f);
        Material duct = Mat("Duct", new Color(0.75f, 0.77f, 0.8f), null, 0.6f);
        Material glass = Mat("Glass", new Color(0.08f, 0.1f, 0.13f), null, 0.85f);
        Material tire = Mat("Tire", new Color(0.06f, 0.06f, 0.06f));
        Material green = Mat("Exit", Color.white, null, 0.0f, new Color(0.1f, 0.9f, 0.3f) * 1.5f);
        Color[] paints = {
            new Color(0.8f, 0.8f, 0.82f), new Color(0.1f, 0.1f, 0.12f), new Color(0.6f, 0.08f, 0.08f),
            new Color(0.1f, 0.2f, 0.5f), new Color(0.45f, 0.47f, 0.5f), new Color(0.9f, 0.9f, 0.88f),
            new Color(0.2f, 0.35f, 0.25f),
        };
        var carPaint = new List<Material>();
        for (int i = 0; i < paints.Length; i++) carPaint.Add(Mat("Car" + i, paints[i], null, 0.75f));

        Shell(H, floor, wall, wallShort, ceiling);
        // Painted band and level sign on the walls
        Box("BandN", Cx, Cy + W / 2 - 0.2f, 1.25f, L, 0.05f, 0.5f, blue);
        Box("BandS", Cx, Cy - W / 2 + 0.2f, 1.25f, L, 0.05f, 0.5f, blue);
        Box("BandW", Cx - L / 2 + 0.2f, Cy, 1.25f, 0.05f, W, 0.5f, blue);
        Box("BandE", Cx + L / 2 - 0.2f, Cy, 1.25f, 0.05f, W, 0.5f, blue);
        Box("Exit", Cx + L / 2 - 0.25f, Cy, 3.0f, 0.1f, 1.4f, 0.5f, green);

        // Ceiling beams across the hall, ducts and pipes along it
        for (float x = Cx - L / 2 + 4; x < Cx + L / 2; x += 8f)
            Box("Beam", x, Cy, H - 0.3f, 0.5f, W, 0.6f, concrete);
        foreach (float y in new[] { Cy - 11f, Cy + 12f })
            Box("Duct", Cx, y, H - 0.9f, L - 1, 0.8f, 0.5f, duct);
        foreach (float y in new[] { Cy - 12.5f, Cy - 12.2f })
            Cyl("Pipe", Cx, y, H - 1.25f, 0.08f, L - 1, pipe, Axis.X);

        // Columns on an 8 m grid outside the track area, with hazard bands
        for (float x = Cx - L / 2 + 4; x < Cx + L / 2; x += 8f)
            for (float y = Cy - W / 2 + 4; y < Cy + W / 2; y += 8f)
            {
                if (x > KeepX0 - 0.5f && x < KeepX1 + 0.5f && y > KeepY0 - 0.5f && y < KeepY1 + 0.5f)
                    continue;
                Box("Column", x, y, H / 2, 0.6f, 0.6f, H, concrete);
                Box("ColumnBand", x, y, 0.6f, 0.62f, 0.62f, 1.2f, hazard);
            }

        // Parking stalls with cars along both long walls, lane lines
        foreach (float s in new[] { -1f, 1f })
        {
            float yStall = Cy + s * (W / 2 - 3.2f);
            for (float x = Cx - L / 2 + 2; x + 2.6f < Cx + L / 2 - 2; x += 2.6f)
            {
                Box("StallLine", x, yStall, 0.005f, 0.12f, 5.0f, 0.01f, white);
                if (rng.NextDouble() < 0.75)
                    Car(x + 1.3f, yStall + s * (float)(rng.NextDouble() * 0.3), s > 0 ? 90f : -90f,
                        carPaint[rng.Next(carPaint.Count)], glass, tire);
            }
            for (float x = Cx - L / 2 + 2; x < Cx + L / 2 - 2; x += 3f)
                Box("LaneDash", x, Cy + s * 12.6f, 0.005f, 1.5f, 0.15f, 0.01f, white);
        }
        FloorRect(KeepX0 + 0.5f, KeepY0 + 0.5f, KeepX1 - 0.5f, KeepY1 - 0.5f, 0.15f, yellow);
        // A few cars parked in the bays west and east of the track area
        for (float y = Cy - 7.5f; y <= Cy + 7.5f; y += 3f)
        {
            Car(Cx - 22f, y, 0f, carPaint[rng.Next(carPaint.Count)], glass, tire);
            if (rng.NextDouble() < 0.6)
                Car(Cx + 21f, y, 180f, carPaint[rng.Next(carPaint.Count)], glass, tire);
        }

        RenderSettings.ambientMode = AmbientMode.Flat;
        RenderSettings.ambientLight = new Color(0.3f, 0.31f, 0.33f);
        for (float x = Cx - L / 2 + 6; x < Cx + L / 2 - 2; x += 6f)
            for (float y = Cy - W / 2 + 5; y < Cy + W / 2 - 2; y += 6f)
                Box("Tube", x, y, H - 0.65f, 0.15f, 2.4f, 0.06f, tube);
        for (float x = Cx - 16; x <= Cx + 16; x += 16)
            foreach (float y in new[] { Cy - 9f, Cy + 9f })
                PointLight(x, y, H - 1.0f, 18f, 1.2f, new Color(0.9f, 0.96f, 1f));
        End(scene, GaragePath);
    }

    // ---- building blocks ----------------------------------------------------

    static void Shell(float h, Material floor, Material wallLong, Material wallShort, Material roof)
    {
        Box("Floor", Cx, Cy, -0.05f, L, W, 0.1f, floor);
        Box("WallN", Cx, Cy + W / 2, h / 2, L, 0.3f, h, wallLong);
        Box("WallS", Cx, Cy - W / 2, h / 2, L, 0.3f, h, wallLong);
        Box("WallW", Cx - L / 2, Cy, h / 2, 0.3f, W, h, wallShort);
        Box("WallE", Cx + L / 2, Cy, h / 2, 0.3f, W, h, wallShort);
        Box("Roof", Cx, Cy, h + 0.15f, L + 0.6f, W + 0.6f, 0.3f, roof);
    }

    static void FloorRect(float x0, float y0, float x1, float y1, float w, Material m)
    {
        Box("Line", (x0 + x1) / 2, y0, 0.005f, x1 - x0, w, 0.01f, m);
        Box("Line", (x0 + x1) / 2, y1, 0.005f, x1 - x0, w, 0.01f, m);
        Box("Line", x0, (y0 + y1) / 2, 0.005f, w, y1 - y0, 0.01f, m);
        Box("Line", x1, (y0 + y1) / 2, 0.005f, w, y1 - y0, 0.01f, m);
    }

    static void RackBay(float x, float y, Material upright, Material beam, Material pallet,
                        Material[] boxes, Random rng)
    {
        const float bay = 2.7f, depth = 1.1f, h = 7.6f;
        foreach (float dx in new[] { -bay / 2, bay / 2 })
            foreach (float dy in new[] { -depth / 2, depth / 2 })
                Box("Upright", x + dx, y + dy, h / 2, 0.09f, 0.09f, h, upright);
        foreach (float z in new[] { 0.15f, 2.5f, 4.85f, 7.2f })
        {
            Box("RackBeam", x, y - depth / 2, z, bay, 0.06f, 0.12f, beam);
            Box("RackBeam", x, y + depth / 2, z, bay, 0.06f, 0.12f, beam);
            if (z > 7f)
                continue;
            for (int k = 0; k < 2; k++)
            {
                if (rng.NextDouble() < 0.15)
                    continue;
                float px = x + (k == 0 ? -0.65f : 0.65f);
                Box("Pallet", px, y, z + 0.13f, 1.1f, 1.0f, 0.14f, pallet);
                float bh = 0.5f + (float)rng.NextDouble() * 1.4f;
                Box("Load", px, y, z + 0.2f + bh / 2, 1.05f, 0.95f, bh, boxes[rng.Next(boxes.Length)]);
            }
        }
    }

    static void PalletStack(float x, float y, Material pallet, Material[] boxes, Random rng)
    {
        float yaw = (float)rng.NextDouble() * 20 - 10;
        int levels = 1 + rng.Next(3);
        float z = 0;
        for (int i = 0; i < levels; i++)
        {
            Box("Pallet", x, y, z + 0.07f, 1.2f, 1.0f, 0.14f, pallet, yaw);
            float bh = 0.6f + (float)rng.NextDouble() * 0.6f;
            Box("Load", x, y, z + 0.14f + bh / 2, 1.15f, 0.95f, bh, boxes[rng.Next(boxes.Length)], yaw);
            z += 0.14f + bh;
        }
    }

    static void Forklift(float x, float y, float yaw, Material body, Material dark, Material steel)
    {
        var g = Group("Forklift", x, y, yaw);
        Box("Body", 0, 0, 0.75f, 2.0f, 1.1f, 0.9f, body, 0, g);
        Box("Counterweight", -0.9f, 0, 0.9f, 0.4f, 1.1f, 0.9f, dark, 0, g);
        Box("Seat", -0.2f, 0, 1.35f, 0.5f, 0.5f, 0.3f, dark, 0, g);
        foreach (float dy in new[] { -0.5f, 0.5f })
        {
            Box("CagePost", 0.45f, dy, 1.9f, 0.06f, 0.06f, 1.4f, dark, 0, g);
            Box("CagePost", -0.6f, dy, 1.9f, 0.06f, 0.06f, 1.4f, dark, 0, g);
            Box("Mast", 1.1f, dy * 0.6f, 1.6f, 0.1f, 0.1f, 3.0f, steel, 0, g);
            Box("Fork", 1.75f, dy * 0.5f, 0.15f, 1.2f, 0.12f, 0.05f, steel, 0, g);
        }
        Box("CageTop", -0.08f, 0, 2.6f, 1.15f, 1.1f, 0.05f, dark, 0, g);
        foreach (float dx in new[] { -0.6f, 0.7f })
            foreach (float dy in new[] { -0.55f, 0.55f })
                Cyl("Wheel", dx, dy, 0.3f, 0.3f, 0.25f, dark, Axis.Y, g);
    }

    static void Car(float x, float y, float yaw, Material paint, Material glass, Material tire)
    {
        var g = Group("Car", x, y, yaw);
        Box("Body", 0, 0, 0.6f, 4.4f, 1.8f, 0.65f, paint, 0, g);
        Box("Cabin", -0.25f, 0, 1.18f, 2.3f, 1.62f, 0.55f, glass, 0, g);
        Box("RoofPanel", -0.3f, 0, 1.47f, 1.9f, 1.55f, 0.04f, paint, 0, g);
        foreach (float dx in new[] { -1.4f, 1.4f })
            foreach (float dy in new[] { -0.8f, 0.8f })
                Cyl("Wheel", dx, dy, 0.33f, 0.33f, 0.24f, tire, Axis.Y, g);
    }

    // ---- primitives (positions and sizes in the ROS frame) -----------------

    enum Axis { X, Y, Z }

    // ROS (x, y, z) -> Unity (x, z, y), as flightlib's positionRos2Unity
    static Vector3 U(float x, float y, float z) => new Vector3(x, z, y);

    static Transform Group(string name, float x, float y, float yawDeg)
    {
        var g = new GameObject(name).transform;
        g.SetParent(root);
        g.position = U(x, y, 0);
        // ROS yaw is counter-clockwise seen from above, Unity's y rotation clockwise
        g.rotation = Quaternion.Euler(0, -yawDeg, 0);
        return g;
    }

    static GameObject Box(string name, float x, float y, float z, float sx, float sy, float sz,
                          Material m, float yawDeg = 0, Transform parent = null)
    {
        var g = GameObject.CreatePrimitive(PrimitiveType.Cube);
        g.name = name;
        g.transform.SetParent(parent ? parent : root, false);
        g.transform.localPosition = U(x, y, z);
        g.transform.localRotation = Quaternion.Euler(0, -yawDeg, 0);
        g.transform.localScale = U(sx, sy, sz);
        return Finish(g, m);
    }

    static GameObject Cyl(string name, float x, float y, float z, float radius, float length,
                          Material m, Axis axis, Transform parent = null)
    {
        var g = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        g.name = name;
        g.transform.SetParent(parent ? parent : root, false);
        g.transform.localPosition = U(x, y, z);
        // The primitive is 2 units long along Unity y, 1 unit in diameter
        g.transform.localScale = new Vector3(2 * radius, length / 2, 2 * radius);
        if (axis == Axis.X)
            g.transform.localRotation = Quaternion.Euler(0, 0, 90);
        else if (axis == Axis.Y)
            g.transform.localRotation = Quaternion.Euler(90, 0, 0);
        return Finish(g, m);
    }

    static GameObject Finish(GameObject g, Material m)
    {
        g.GetComponent<MeshRenderer>().sharedMaterial = m;
        // Visual only: the ROS simulator has no collisions with the scenery
        Object.DestroyImmediate(g.GetComponent<Collider>());
        g.isStatic = true;
        return g;
    }

    static void PointLight(float x, float y, float z, float range, float intensity, Color c)
    {
        var g = new GameObject("Light");
        g.transform.SetParent(root);
        g.transform.position = U(x, y, z);
        var l = g.AddComponent<Light>();
        l.type = LightType.Point;
        l.range = range;
        l.intensity = intensity;
        l.color = c;
        l.shadows = LightShadows.None;
        l.renderMode = LightRenderMode.ForcePixel;
    }

    // ---- scene, material and texture assets --------------------------------

    static Scene Begin(string scenePath, bool keepSun)
    {
        dir = Path.GetDirectoryName(scenePath).Replace('\\', '/');
        // Regenerate from scratch so the scene always matches this script
        if (AssetDatabase.IsValidFolder(dir))
            AssetDatabase.DeleteAsset(dir);
        AssetDatabase.CreateFolder(Path.GetDirectoryName(dir).Replace('\\', '/'), Path.GetFileName(dir));

        Scene scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
        // Flightmare spawns its own cameras
        Object.DestroyImmediate(GameObject.Find("Main Camera"));
        var sun = GameObject.Find("Directional Light");
        if (keepSun)
        {
            sun.GetComponent<Light>().shadows = LightShadows.Soft;
            sun.transform.rotation = Quaternion.Euler(50, -30, 0);
        }
        else
        {
            Object.DestroyImmediate(sun);
        }
        root = new GameObject("Environment").transform;
        return scene;
    }

    static void End(Scene scene, string path)
    {
        EditorSceneManager.SaveScene(scene, path);
    }

    static Material Mat(string name, Color color, Texture2D tex = null, float smoothness = 0.15f,
                        Color? emission = null)
    {
        var m = new Material(Shader.Find("Standard"));
        m.color = color;
        if (tex != null)
            m.mainTexture = tex;
        m.SetFloat("_Glossiness", smoothness);
        if (emission.HasValue)
        {
            m.EnableKeyword("_EMISSION");
            m.SetColor("_EmissionColor", emission.Value);
            m.globalIlluminationFlags = MaterialGlobalIlluminationFlags.None;
        }
        AssetDatabase.CreateAsset(m, dir + "/" + name + ".mat");
        return m;
    }

    static Texture2D Tex(string name, int size, Func<int, int, Color> pixel)
    {
        string path = dir + "/" + name + ".png";
        if (!File.Exists(path))
        {
            var t = new Texture2D(size, size, TextureFormat.RGB24, false);
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                    t.SetPixel(x, y, pixel(x, y));
            File.WriteAllBytes(path, t.EncodeToPNG());
            AssetDatabase.ImportAsset(path);
            var importer = (TextureImporter)AssetImporter.GetAtPath(path);
            importer.wrapMode = TextureWrapMode.Repeat;
            importer.SaveAndReimport();
        }
        return AssetDatabase.LoadAssetAtPath<Texture2D>(path);
    }

    // Mottled concrete: three octaves of Perlin noise, blended across the
    // borders so the texture tiles without seams
    static Texture2D Concrete(string name, float contrast)
    {
        const int n = 256;
        return Tex(name, n, (x, y) =>
        {
            float v = 0, amp = 0.5f;
            for (int o = 0; o < 3; o++, amp *= 0.5f)
                v += amp * TiledPerlin(x, y, n, 0.035f * (1 << o), 17.3f * o);
            float g = 1 - contrast * 0.5f + contrast * v;
            return new Color(g, g, g * 0.98f);
        });
    }

    static float TiledPerlin(int x, int y, int n, float freq, float seed)
    {
        float u = (float)x / n, w = (float)y / n;
        float P(float px, float py) => Mathf.PerlinNoise(seed + px * freq, seed + py * freq);
        return P(x, y) * (1 - u) * (1 - w) + P(x - n, y) * u * (1 - w)
             + P(x, y - n) * (1 - u) * w + P(x - n, y - n) * u * w;
    }

    // Vertical ribs of corrugated metal sheet
    static Texture2D Corrugated(string name, int ribs = 8)
    {
        return Tex(name, 128, (x, y) =>
        {
            float g = 0.82f + 0.18f * Mathf.Sin(x / 128f * Mathf.PI * 2 * ribs);
            return new Color(g, g, g);
        });
    }

    static Texture2D Hazard(string name)
    {
        return Tex(name, 128, (x, y) => ((x + y) / 32) % 2 == 0
            ? new Color(0.95f, 0.78f, 0.05f) : new Color(0.08f, 0.08f, 0.08f));
    }
}
