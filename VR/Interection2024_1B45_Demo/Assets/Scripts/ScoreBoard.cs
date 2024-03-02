using System;
using System.Linq;
using System.Text;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using System.Net;
using System.Net.Sockets;
using System.Threading;

using TMPro;

public class ScoreBoard : MonoBehaviour
{
    [SerializeField] TextMeshProUGUI score_text;
    private int score;
    private int max_combo;
    private ReceivedTapData receivedTapData;
    Thread receiveThread;
    UdpClient client;
    private int port;

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
    // Start is called before the first frame update
    void Start()
    {
        score = PlayerPrefs.GetInt("SCORE");
        max_combo = PlayerPrefs.GetInt("MAX COMBO");
        score_text.text = "Congratulations!\n";
        score_text.text += "Score: " + score.ToString() + "\n";
        score_text.text += "MAX COMBO: " + max_combo.ToString();
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    // Update is called once per frame
    void Update()
    {
        if (receivedTapData != null)
        {
            if (receivedTapData.is_tapped_l)
            {
                // Back to title
                if (receivedTapData.tapped_part_l == "LeftHip")
                {
                    receiveThread.Abort();
                    Invoke("BackToTitle", 1);
                }

                // Retry
                if (receivedTapData.tapped_part_l == "RightHip")
                {
                    receiveThread.Abort();
                    Invoke("Retry", 1);
                }
            }

            if (receivedTapData.is_tapped_r)
            {
                // Back to title
                if (receivedTapData.tapped_part_r == "LeftHip")
                {
                    receiveThread.Abort();
                    Invoke("BackToTitle", 1);
                }

                // Retry
                if (receivedTapData.tapped_part_r == "RightHip")
                {
                    receiveThread.Abort();
                    Invoke("Retry", 1);
                }
            }
        }
    }
    
    void BackToTitle()
    {
        SceneManager.LoadScene("Title");
    }

    void Retry()
    {
        SceneManager.LoadScene("DemoApp");
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
