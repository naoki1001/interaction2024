using System;
using System.Linq;
using System.Text;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Threading;

using TMPro;

public class DemoManager : MonoBehaviour
{
    [SerializeField] TextMeshProUGUI score_text;
    [SerializeField, EnumIndex(typeof(JointName))] private GameObject[] TapPositionAnchor;
    // [SerializeField, EnumIndex(typeof(JointName))] private Transform[] TrackPointReferences;
    int score = 0;
    int combo = 0;
    private NotesCollisionManager notesCollisionManager;

    /// <summary>
    /// 各インデックスのジョイントの名前
    /// </summary>
    public enum JointName
    {
        LeftHip = 0,
        RightHip,
        LeftShoulder,
        RightShoulder,
        Head
    }

    /// <summary>
    /// ジョイントの総数
    /// </summary>
    private int JointNumber => System.Enum.GetValues(typeof(JointName)).Length;
    private ReceivedTapData receivedTapData;
    Thread receiveThread;
    UdpClient client;
    private int port;
    private bool is_tapped_l = false;
    private bool is_tapped_r = false;
    private float tap_position_z;

    [System.Serializable]
    public class ReceivedTapData
    {
        public bool is_tapped_l;
        public string tapped_part_l;
        public bool is_tapped_r;
        public string tapped_part_r;
        public ReceivedTapData()
        {
            this.is_tapped_l = false;
            this.is_tapped_r = false;
            this.tapped_part_l = null;
            this.tapped_part_r = null;
        }
    }

    // // 音源再生
    // private AudioSource audio;

    // // 効果音の格納
    // [SerializeField]
    // private AudioClip soundSE;

    void Start()
    {
        score_text.text = "Score: 0";
        tap_position_z = TapPositionAnchor[0].transform.position.z;
        // audio = gameObject.AddComponent<AudioSource>();
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    // Update is called once per frame
    void Update()
    {
        score_text.text = "Score: " + score.ToString() + "\n";
        score_text.text += "Combo: " + combo.ToString();
        if (receivedTapData != null)
        {
            if (receivedTapData.is_tapped_l && !is_tapped_l && receivedTapData.tapped_part_l != null)
            {
                notesCollisionManager = TapPositionAnchor[(int)(JointName)Enum.Parse(typeof(JointName), receivedTapData.tapped_part_l)].GetComponent<NotesCollisionManager>();
                float diff = Mathf.Abs(notesCollisionManager.area.z - tap_position_z);
                if (notesCollisionManager.inLine)
                {
                    is_tapped_l = true;
                    receivedTapData.is_tapped_l = false;
                    if (diff < 0.5f)
                    {
                        score += (int)(20.0f * (1.0f + (float)combo / 10.0f));
                    }
                    else
                    {
                        score += (int)(10.0f * (1.0f + (float)combo / 10.0f));
                    }
                    combo++;
                    Destroy(notesCollisionManager.note);
                }
                if (diff > 1.0f)
                {
                    if (notesCollisionManager.note)
                    {
                        combo = 0;
                        Destroy(notesCollisionManager.note);
                    }
                }
            }
            else if (!receivedTapData.is_tapped_l || receivedTapData.tapped_part_l == null)
            {
                is_tapped_l = false;
            }

            if (receivedTapData.is_tapped_r && !is_tapped_r && receivedTapData.tapped_part_r != null)
            {
                notesCollisionManager = TapPositionAnchor[(int)(JointName)Enum.Parse(typeof(JointName), receivedTapData.tapped_part_r)].GetComponent<NotesCollisionManager>();
                float diff = Mathf.Abs(notesCollisionManager.area.z - tap_position_z);
                if (notesCollisionManager.inLine)
                {
                    is_tapped_r = true;
                    receivedTapData.is_tapped_r = false;
                    if (diff < 0.5f)
                    {
                        score += (int)(20.0f * (1.0f + (float)combo / 10.0f));
                    }
                    else
                    {
                        score += (int)(10.0f * (1.0f + (float)combo / 10.0f));
                    }
                    combo++;
                    Destroy(notesCollisionManager.note);
                }
                if (diff > 1.0f)
                {
                    if (notesCollisionManager.note)
                    {
                        combo = 0;
                        Destroy(notesCollisionManager.note);
                    }
                }
            }
            else if (!receivedTapData.is_tapped_r || receivedTapData.tapped_part_r == null)
            {
                is_tapped_r = false;
            }
        }
    }

    void OnApplicationQuit()
    {
        if (receiveThread != null) receiveThread.Abort();
    }

    // receive thread
    private void ReceiveData()
    {
        port = 8083;
        client = new UdpClient(port);
        print("Starting Server");
        while (true)
        {
            try
            {
                IPEndPoint remoteEndPoint = new IPEndPoint(IPAddress.Parse("127.0.0.1"), port);
                byte[] data = client.Receive(ref remoteEndPoint);
                string message = Encoding.UTF8.GetString(data);
                Debug.Log("Received: " + message);
                receivedTapData = JsonUtility.FromJson<ReceivedTapData>(message);
            }
            catch (Exception err)
            {
                print(err.ToString());
            }
        }
    }
}
