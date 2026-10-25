// Generates an empty racing arena scene for the Flightmare renderer: a 300 m
// checkered ground (1 m squares, for a sense of speed and depth), sun and sky,
// nothing else. Gates come from the static_obstacles.csv that the race overlay
// writes (acmpc/race_track.py).
// Copied into <flightmare_unity>/Assets/Editor/ by scripts/build_renderer.sh;
// BuildMacOS.Build calls RaceArena.Ensure() and adds the scene to the build.
// build_renderer.sh registers it in sceneSchedule.cs as scene_id 4.
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;

public static class RaceArena
{
    public const string Dir = "Assets/Environments/Arena";
    public const string ScenePath = Dir + "/Arena.unity";

    public static void Ensure()
    {
        if (File.Exists(ScenePath))
            return;
        Directory.CreateDirectory(Dir);

        // Two-tone checker texture, 2x2 cells per repeat
        string texPath = Dir + "/checker.png";
        var tex = new Texture2D(256, 256);
        for (int y = 0; y < 256; y++)
            for (int x = 0; x < 256; x++)
                tex.SetPixel(x, y, ((x / 128) + (y / 128)) % 2 == 0
                    ? new Color(0.62f, 0.64f, 0.66f) : new Color(0.42f, 0.44f, 0.46f));
        File.WriteAllBytes(texPath, tex.EncodeToPNG());
        AssetDatabase.ImportAsset(texPath);

        var mat = new Material(Shader.Find("Standard"));
        mat.mainTexture = AssetDatabase.LoadAssetAtPath<Texture2D>(texPath);
        // Plane: 10 units * 30 = 300 m; 150 repeats of 2 m -> 1 m cells
        mat.mainTextureScale = new Vector2(150, 150);
        mat.SetFloat("_Glossiness", 0.1f);
        AssetDatabase.CreateAsset(mat, Dir + "/Ground.mat");

        Scene scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
        // Flightmare spawns its own cameras
        Object.DestroyImmediate(GameObject.Find("Main Camera"));
        var sun = GameObject.Find("Directional Light").GetComponent<Light>();
        sun.shadows = LightShadows.Soft;
        sun.transform.rotation = Quaternion.Euler(50, -30, 0);

        GameObject ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
        ground.name = "Ground";
        ground.transform.localScale = new Vector3(30, 1, 30);
        ground.GetComponent<MeshRenderer>().sharedMaterial = mat;

        RenderSettings.ambientMode = AmbientMode.Skybox;
        EditorSceneManager.SaveScene(scene, ScenePath);
        AssetDatabase.SaveAssets();
    }
}
