using UnityEngine;

public class AIDragHandler : MonoBehaviour
{
    private Camera mainCamera;
    private Rigidbody rb;
    private Animator animator;
    private bool isDragging = false;
    private Vector3 offset;
    private Plane dragPlane;
    public float dragSpeed = 10f;
    private float groundedCheckTimer = 0f;
    private float groundedGraceTime = 0.1f;
    private float groundRaycastDistance = 1.0f;
    private float groundedThreshold = 0.1f;

    void Start()
    {
        mainCamera = Camera.main;
        rb = GetComponent<Rigidbody>();
        animator = GetComponent<Animator>();
        if (!rb)
        {
            Debug.LogError("Rigidbody is required on the character for physics dragging!");
        }
        dragPlane = new Plane(-mainCamera.transform.forward, transform.position);
    }

    void Update()
    {
        HandleMouseInput();
        CheckGroundedStatus();
    }

    void HandleMouseInput()
    {
        if (Input.GetMouseButtonDown(0))
        {
            Ray ray = mainCamera.ScreenPointToRay(Input.mousePosition);
            int layerMask = LayerMask.GetMask("AuroraRoot"); // Only select AuroraRoot
            if (Physics.Raycast(ray, out RaycastHit hit, 100f, layerMask))
            {
                if (hit.transform == transform)
                {
                    StartDragging(hit.point); // ✅ Immediate fast drag trigger
                }
            }
        }

        if (Input.GetMouseButton(0) && isDragging)
        {
            Drag();
        }

        if (Input.GetMouseButtonUp(0) && isDragging)
        {
            StopDragging();
        }
    }


    void StartDragging(Vector3 hitPoint)
    {
        if (animator)
        {
            animator.applyRootMotion = false;    
            animator.SetBool("Grounded", false); 
            animator.SetBool("IsDragged", true);  // ✅ Tell Animator we're dragging
        }
        dragPlane = new Plane(-mainCamera.transform.forward, transform.position);
        offset = transform.position - hitPoint;
        isDragging = true;


    }

    void StopDragging()
    {
        rb.velocity = Vector3.zero; // Clean stop
        animator.SetBool("IsDragged", false);
        isDragging = false;
    }


    void Drag()
    {
        Ray ray = mainCamera.ScreenPointToRay(Input.mousePosition);
        if (dragPlane.Raycast(ray, out float distance))
        {
            Vector3 worldPosition = ray.GetPoint(distance);
            Vector3 targetPosition = worldPosition + offset;

            Vector3 moveDirection = (targetPosition - transform.position) * dragSpeed;
            rb.velocity = moveDirection;
        }
    }
    
    void CheckGroundedStatus()
    {
        if (isDragging) return; // Don't check while dragging

        Ray groundRay = new Ray(transform.position, Vector3.down);
        bool isGroundDetected = Physics.Raycast(groundRay, out RaycastHit hit, groundRaycastDistance);

        if (isGroundDetected)
        {
            float groundDistance = hit.distance;

            if (groundDistance <= groundedThreshold)
            {
                // ✅ Snap immediately to grounded if touching close enough
                if (!animator.GetBool("Grounded"))
                {
                    animator.SetBool("Grounded", true);
                    
                    animator.applyRootMotion = true;
                }
                groundedCheckTimer = 0f; // ✅ Reset timer when touching ground
            }
            else
            {
                groundedCheckTimer += Time.deltaTime;
                if (groundedCheckTimer >= groundedGraceTime)
                {
                    if (animator.GetBool("Grounded"))
                    {
                        animator.SetBool("Grounded", false);
                    }
                }
            }
        }
        else
        {
            groundedCheckTimer += Time.deltaTime;
            if (groundedCheckTimer >= groundedGraceTime)
            {
                if (animator.GetBool("Grounded"))
                {
                    animator.SetBool("Grounded", false);
                }
            }
        }
    }

}
